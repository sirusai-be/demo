Command options
================
1) run with options specified
uv run sora2_generate.py "A cinematic shot of a sunset over the mountains." --model sora-2 --size 1280x720 --seconds 4 --download-path sdsunsest.mp4

2) run with a prompt file specified
uv run sora2_generate.py --prompt-file sora_prompt.txt

3) run to pull/download without regenerating
uv run python .\sora2_generate.py --resume video_68e543f4db08819190e87c2bcc96fea608ac2a8eab7cd858 .\video.mp4 --debug

4) run with image (match to size)
uv run sora2_generate.py --match-size-to-input --input-reference "@jan4.png;type=image/png"
