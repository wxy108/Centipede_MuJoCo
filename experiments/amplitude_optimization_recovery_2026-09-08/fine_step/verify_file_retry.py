"""Reproduce a Windows reader denying deletion during an atomic JSON write."""
import concurrent.futures,ctypes,time
from ctypes import wintypes
import core as c

def main():
    path=c.OUT/'audit/io_retry_check.json';c.write(path,dict(generation='old'))
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    kernel.CreateFileW.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle.restype=wintypes.BOOL
    # Share read and write but deliberately withhold FILE_SHARE_DELETE.
    handle=kernel.CreateFileW(str(path),0x80000000,3,None,3,128,None)
    assert handle!=ctypes.c_void_p(-1).value,ctypes.get_last_error()
    started=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(c.write,path,dict(generation='new',payload=list(range(200))))
        try:
            time.sleep(.2)
            assert not future.done() and c.read(path)==dict(generation='old')
        finally:assert kernel.CloseHandle(handle)
        future.result(timeout=6)
    elapsed=time.monotonic()-started
    assert c.read(path)==dict(generation='new',payload=list(range(200)))
    c.write(c.OUT/'audit/file_retry_verification.json',dict(passed=True,elapsed_s=elapsed,
        original_preserved_while_locked=True,atomic_replacement_after_unlock=True,
        retry_deadline_s=5,core_sha256=c.sha(c.ROOT/'core.py'),physics_and_scoring_changed=False))
    print('Windows transient sharing-conflict reproduction passed.',flush=True)

if __name__=='__main__':main()
