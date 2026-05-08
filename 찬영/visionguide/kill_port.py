"""포트 8000을 사용하는 모든 프로세스 강제 종료"""
import subprocess, sys

result = subprocess.run(
    ["netstat", "-ano"],
    capture_output=True, text=True
)

pids = set()
for line in result.stdout.splitlines():
    if ":8000" in line and ("LISTENING" in line or "ESTABLISHED" in line):
        parts = line.split()
        if parts:
            pids.add(parts[-1])

print(f"포트 8000 관련 PID: {pids}")

for pid in pids:
    if pid == "0":
        continue
    r = subprocess.run(["taskkill", "/PID", pid, "/F", "/T"],
                       capture_output=True, text=True)
    print(f"PID {pid}: {r.stdout.strip() or r.stderr.strip()}")

print("완료. 이제 서버를 다시 시작하세요.")
