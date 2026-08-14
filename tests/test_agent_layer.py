"""Tests for Phase C: the agent-facing layer.

Covers the `similar` field on store responses (near-duplicate flagging with
the mid-write agent as judge), the consolidation_report tool (duplicate
clusters from stored vectors + stale never-retrieved items), the clustering
helper, and the consolidate/briefing prompts.
"""

import unittest
from unittest.mock import patch

import mnemomatic.server as server
from mnemomatic import config
from mnemomatic.db import Database
from mnemomatic.models import Knowledge, Note
from mnemomatic import runtime
from mnemomatic import tools_content
from mnemomatic import tools_history
from tests._support import axis, mix


class ToolTestCase(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self._patches = [
            patch.object(runtime, "_db", return_value=self.db),
            patch.object(runtime, "_embedder", return_value=None),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self.db.close()


class TestSimilarOnStore(ToolTestCase):
    def _store_with_vector(self, title, vector, content="body"):
        with patch.object(runtime, "_safe_embed", return_value=vector):
            return tools_content.store_note(namespace="proj", title=title, content=content)

    def test_near_duplicate_is_flagged(self):
        first = self._store_with_vector("original", axis(0))
        second = self._store_with_vector("copycat", mix(0, 1, 0.95, 0.31))  # cos ~0.95
        self.assertIn("similar", second)
        self.assertEqual(second["similar"][0]["id"], first["id"])
        self.assertGreaterEqual(second["similar"][0]["score"], 0.9)

    def test_distinct_content_is_not_flagged(self):
        self._store_with_vector("original", axis(0))
        second = self._store_with_vector("unrelated", axis(1))
        self.assertNotIn("similar", second)

    def test_item_never_flags_itself(self):
        first = self._store_with_vector("only one", axis(0))
        self.assertNotIn("similar", first)
        # Re-storing the same item (upsert) must not flag itself either.
        again = self._store_with_vector("only one", axis(0), content="body v2")
        self.assertNotIn("similar", again)

    def test_no_embedding_no_flag(self):
        tools_content.store_note(namespace="proj", title="a", content="x")
        second = tools_content.store_note(namespace="proj", title="b", content="x")
        self.assertNotIn("similar", second)

    def test_threshold_zero_disables(self):
        self._store_with_vector("original", axis(0))
        with patch.object(config, "SIMILAR_THRESHOLD", 0):
            second = self._store_with_vector("copy", axis(0))
        self.assertNotIn("similar", second)

    def test_other_namespaces_do_not_flag(self):
        self._store_with_vector("original", axis(0))
        with patch.object(runtime, "_safe_embed", return_value=axis(0)):
            second = tools_content.store_note(namespace="elsewhere", title="same", content="body")
        self.assertNotIn("similar", second)

    def test_knowledge_store_flags_too(self):
        with patch.object(runtime, "_safe_embed", return_value=axis(0)):
            first = tools_content.store_knowledge(namespace="proj", subject="s1", fact="f1")
            second = tools_content.store_knowledge(namespace="proj", subject="s2", fact="f2")
        self.assertEqual([s["id"] for s in second["similar"]], [first["id"]])


class TestDuplicateClusters(unittest.TestCase):
    def test_clusters_and_scores(self):
        vectors = [
            ("a", "A", axis(0)),
            ("b", "B", mix(0, 1, 0.95, 0.31)),   # ~0.95 to a
            ("c", "C", axis(2)),                  # unrelated
            ("d", "D", axis(3)),
            ("e", "E", axis(3)),                  # identical to d
        ]
        clusters = tools_history._duplicate_clusters("note", vectors, threshold=0.8)
        clusters.sort(key=lambda c: c["similarity"])
        self.assertEqual(len(clusters), 2)
        self.assertEqual({i["id"] for i in clusters[0]["items"]}, {"a", "b"})
        self.assertEqual({i["id"] for i in clusters[1]["items"]}, {"d", "e"})
        self.assertEqual(clusters[1]["similarity"], 1.0)

    def test_transitive_chain_becomes_one_cluster(self):
        # a~b and b~c above threshold, a~c below: union-find still groups them.
        vectors = [
            ("a", "A", mix(0, 1, 1.0, 0.0)),
            ("b", "B", mix(0, 1, 0.9, 0.44)),
            ("c", "C", mix(0, 1, 0.62, 0.78)),
        ]
        clusters = tools_history._duplicate_clusters("note", vectors, threshold=0.85)
        self.assertEqual(len(clusters), 1)
        self.assertEqual({i["id"] for i in clusters[0]["items"]}, {"a", "b", "c"})

    def test_empty_and_singleton(self):
        self.assertEqual(tools_history._duplicate_clusters("note", [], 0.8), [])
        self.assertEqual(tools_history._duplicate_clusters("note", [("a", "A", axis(0))], 0.8), [])


class TestConsolidationReport(ToolTestCase):
    def test_report_shape_clusters_and_stale(self):
        self.db.store_knowledge(Knowledge(namespace="proj", subject="s1", fact="f1"), axis(0))
        self.db.store_knowledge(Knowledge(namespace="proj", subject="s2", fact="f2"), axis(0))
        self.db.store_note(Note(namespace="proj", title="lonely", content="x"), axis(5))

        report = tools_history.consolidation_report(namespace="proj", stale_days=0)
        self.assertEqual(len(report["duplicate_clusters"]), 1)
        cluster = report["duplicate_clusters"][0]
        self.assertEqual(cluster["type"], "knowledge")
        self.assertEqual({i["title"] for i in cluster["items"]}, {"s1", "s2"})
        # Everything has retrieval_count 0 and stale_days=0, so all are stale.
        self.assertEqual(len(report["stale"]), 3)
        self.assertEqual(report["counts"]["knowledge"], 2)

    def test_retrieved_items_are_not_stale(self):
        note, _ = self.db.store_note(Note(namespace="proj", title="used", content="x"), None)
        self.db.store_note(Note(namespace="proj", title="unused", content="y"), None)
        self.db.record_access([("note", note.id)])
        report = tools_history.consolidation_report(namespace="proj", stale_days=0)
        self.assertEqual([r["title"] for r in report["stale"]], ["unused"])

    def test_superseded_facts_do_not_cluster(self):
        self.db.store_knowledge(Knowledge(namespace="proj", subject="s", fact="old"), axis(0))
        self.db.store_knowledge(Knowledge(namespace="proj", subject="s", fact="new"), axis(0))
        report = tools_history.consolidation_report(namespace="proj", stale_days=0)
        # The superseded row lost its vector; only the current fact remains.
        self.assertEqual(report["duplicate_clusters"], [])

    def test_invalid_threshold(self):
        self.assertIn("error", tools_history.consolidation_report(namespace="proj",
                                                           similarity_threshold=0))

    def test_empty_namespace(self):
        report = tools_history.consolidation_report(namespace="nothing-here")
        self.assertEqual(report["duplicate_clusters"], [])
        self.assertEqual(report["stale"], [])
        self.assertEqual(report["counts"], {})


class TestPrompts(unittest.TestCase):
    def test_consolidate_prompt_mentions_the_workflow(self):
        text = tools_history.consolidate("proj")
        self.assertIn("consolidation_report(namespace='proj')", text)
        for tool in ("read()", "update_knowledge", "list_revisions"):
            self.assertIn(tool, text)

    def test_briefing_prompt_embeds_task_and_scope(self):
        text = tools_history.briefing("upgrade the auth flow", namespace="webapp")
        self.assertIn("upgrade the auth flow", text)
        self.assertIn("namespace='webapp'", text)
        for tool in ("search()", "read()", "fact_history"):
            self.assertIn(tool, text)

    def test_briefing_prompt_global_scope(self):
        text = tools_history.briefing("some task")
        self.assertIn("whole store", text)

    def test_prompts_are_registered(self):
        import asyncio
        prompts = asyncio.run(server.mcp.list_prompts())
        names = {p.name for p in prompts}
        self.assertEqual({"consolidate", "briefing"} & names, {"consolidate", "briefing"})


if __name__ == "__main__":
    unittest.main()
