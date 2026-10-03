"""Offline recovery commands that touch the identity tables directly.

    python -m mnemomatic.admin_cli create-admin <username>
    python -m mnemomatic.admin_cli reset-password <username>

For when nobody can log in any more: the only admin forgot their password,
or was deactivated by mistake. Runs inside the container against the same
database file the server uses (``docker exec <container> /usr/bin/python3 -m
mnemomatic.admin_cli ...``), prints a temporary password, and leaves an audit
row attributed to ``cli``. Everything else is done in the web UI.
"""

import argparse
import sys

from mnemomatic import config
from mnemomatic.db import Database
from mnemomatic.audit import write_event
from mnemomatic.identity import Identity, IdentityError


def _open() -> tuple[Database, Identity]:
    db = Database(config.DB_PATH, allow_reindex=True)
    return db, Identity(db)


def cmd_create_admin(args) -> int:
    db, identity = _open()
    try:
        user, temp = identity.create_user(args.username, role="admin", display_name=args.display_name)
    except IdentityError as e:
        print(f"error: {e.details}", file=sys.stderr)
        return 1
    write_event(db, "admin.created", actor="cli", item_type="user", item_id=user.username, source="cli")
    print(f"Created admin {user.username!r}.")
    print(f"Temporary password (valid 7 days, must be changed at first login): {temp}")
    return 0


def cmd_reset_password(args) -> int:
    db, identity = _open()
    user = identity.get_user_by_name(args.username)
    if user is None:
        print(f"error: no user named {args.username!r}", file=sys.stderr)
        return 1
    # acting_user_id=0 can never equal a real id, so the self-reset guard
    # (meant for the web UI) does not apply to an operator at the console.
    temp, expires = identity.reset_password(user.id, acting_user_id=0)
    if not user.active:
        with db.write() as conn:
            conn.execute("UPDATE users SET active = 1 WHERE id = ?", (user.id,))
        print(f"Reactivated {user.username!r}.")
    write_event(db, "password.reset", actor="cli", item_type="user", item_id=user.username,
                source="cli", expires_at=expires)
    print(f"Temporary password for {user.username!r} (valid until {expires}): {temp}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mnemomatic.admin_cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("create-admin", help="create an administrator with a temporary password")
    p.add_argument("username")
    p.add_argument("--display-name", default="Administrator")
    p.set_defaults(func=cmd_create_admin)
    p = sub.add_parser("reset-password", help="give a user a temporary password (and reactivate them)")
    p.add_argument("username")
    p.set_defaults(func=cmd_reset_password)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
