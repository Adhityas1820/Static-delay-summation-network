import subprocess
from pathlib import Path
from collections import defaultdict

def get_video_info(path: Path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", 
           "-show_entries", "stream=codec_name", 
           "-show_entries", "format=duration",
           "-of", "csv=p=0", str(path)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
        # output is usually like: codec,duration
        parts = res.split(',')
        if len(parts) >= 2:
            return parts[0].strip(), parts[1].strip()
    except Exception:
        pass
    return None, None

def main():
    videos_dir = Path(r"C:\Adhitya\Coding\test\NewNN\videos")
    if not videos_dir.exists():
        print(f"Directory {videos_dir} not found.")
        return

    # Group videos by duration (rounded to 1 decimal place, e.g. 18.2 seconds)
    duration_groups = defaultdict(list)
    
    print("Scanning videos...")
    files = [f for f in videos_dir.iterdir() if f.is_file() and f.suffix.lower() in [".mp4", ".mkv", ".mov", ".webm"]]
    
    for i, file in enumerate(files):
        safe_name = file.name.encode('ascii', 'replace').decode('ascii')
        print(f"[{i+1}/{len(files)}] Probing {safe_name}...")
        codec, duration = get_video_info(file)
        if codec and duration:
            try:
                dur_float = round(float(duration), 1)
                duration_groups[dur_float].append({"path": file, "codec": codec})
            except ValueError:
                pass

    removed_count = 0
    for dur, items in duration_groups.items():
        if len(items) > 1:
            # We found videos with the exact same duration!
            has_h264 = any(i["codec"] in ["h264", "hevc"] for i in items)
            
            # If we have an H.264 version, delete any AV1 versions of the same duration
            if has_h264:
                for item in items:
                    if item["codec"] == "av1":
                        print(f"[Duplicate Found] Kept H.264 version. Deleting AV1 duplicate: {item['path'].name}")
                        item["path"].unlink(missing_ok=True)
                        removed_count += 1

    print(f"\nScan complete! Deleted {removed_count} redundant AV1 duplicates.")

if __name__ == "__main__":
    main()
