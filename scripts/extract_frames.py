
import argparse, os, cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--timestamps", nargs="+", type=float, required=True)
    ap.add_argument("--outdir", default="frames")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[info] fps={fps}, total={total}, dur={total/fps:.1f}s")
    for t in args.timestamps:
        fno = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ok, fr = cap.read()
        if not ok:
            print(f"[warn] no frame at t={t}")
            continue
        out = os.path.join(args.outdir, f"frame_t{t:07.2f}_f{fno}.png")
        cv2.imwrite(out, fr)
        print(f"[ok] {out}  shape={fr.shape}")
    cap.release()

if __name__ == "__main__":
    main()