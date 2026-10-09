import argparse

from .export import export_charts


def main(argv=None):
    ap = argparse.ArgumentParser(prog="camera-iq")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export-charts", help="write print-ready PDF + PNG of both charts (US Letter landscape, 100%% scale)")
    ex.add_argument("outdir")
    ex.add_argument("--dpi", type=int, default=300, help="PNG resolution (default 300)")
    args = ap.parse_args(argv)
    if args.cmd == "export-charts":
        for p in export_charts(args.outdir, args.dpi):
            print(p)


if __name__ == "__main__":
    main()
