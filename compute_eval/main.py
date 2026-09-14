import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from compute_eval.generation.main import main as completion_generator_main
from compute_eval.harbor.cli import main as harbor_main


def main(argv: list[str] | None = None) -> None:
    # Explicit path: load_dotenv's default search walks from the caller's
    # source file, which lands in site-packages when installed as a package.
    load_dotenv(Path.cwd() / ".env", override=False, interpolate=True)

    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["harbor"]:
        # Pass the "harbor" token into the sub-parser's prog so its usage/help
        # reads "<prog> harbor generate ..." rather than dropping the subcommand.
        prog = f"{os.path.basename(sys.argv[0])} harbor"
        harbor_main(args[1:], prog=prog)
        return
    completion_generator_main(args)


if __name__ == "__main__":
    main()
