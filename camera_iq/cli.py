import argparse

from .config import DEFAULT, Thresholds
from .export import export_charts


def main(argv=None):
    ap = argparse.ArgumentParser(prog="camera-iq")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export-charts", help="write print-ready PDF + PNG of both charts (US Letter landscape, 100%% scale)")
    ex.add_argument("outdir")
    ex.add_argument("--dpi", type=int, default=300, help="PNG resolution (default 300)")
    an = sub.add_parser("analyze", help="analyse captures; the first is the baseline the rest are compared against")
    an.add_argument("captures", nargs="+", help="image files, or folders (one folder = one capture made of several images)")
    an.add_argument("-o", "--output", default="report.html")
    an.add_argument("--mtf-drop", type=float, default=DEFAULT.mtf50_drop, help="MTF50 drop fraction that flags (default %(default)s)")
    an.add_argument("--de-rise", type=float, default=DEFAULT.delta_e_rise, help="mean dE00 rise that flags (default %(default)s)")
    args = ap.parse_args(argv)
    if args.cmd == "export-charts":
        for p in export_charts(args.outdir, args.dpi):
            print(p)
    elif args.cmd == "analyze":
        return _analyze(args)


def _analyze(args):
    from .pipeline import analyze_path, compare
    from .report import render_report
    th = Thresholds(args.mtf_drop, args.de_rise)
    results = [analyze_path(p) for p in args.captures]
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(render_report(results, th))
    bad = 0
    for r in results[1:]:
        for fl in compare(results[0], r, th):
            print(f"REGRESSION {r.name}: {fl.message}")
            bad += 1
    for r in results:
        for n in r.notes:
            print(f"note {r.name}: {n}")
    print(f"report written to {args.output}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
