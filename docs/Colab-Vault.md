# Encrypted TextGen in Colab

[Open the notebook](https://colab.research.google.com/github/NulAsh/textgen/blob/main/Colab-Encrypted-TextGen.ipynb).
It runs the full TextGen interface inside Colab, with the same browser password
unlock and vault format as the local fork. The ordinary local launcher keeps
its existing Landlock behavior.

This is experimental support. A real Colab session is needed to validate its
current browser proxy, DriveFS behavior and GPU driver. The notebook includes
a mandatory isolation preflight and stops if it fails. Passing Linux CI is
not evidence that every managed Colab runtime exposes the same facilities.

## Run

1. Select **Runtime → Change runtime type → GPU**.
2. Run cell 1. It installs this fork, an isolated Python environment and
   bubblewrap, then tests actual filesystem confinement before opening a vault.
3. Run cell 2 to install the CUDA portable requirements. Enter a direct public
   HTTPS GGUF URL, or upload your model to `/content/textgen-models` using the
   Colab Files panel. Model weights remain outside the vault.
4. Run cell 3 and authorize Google Drive mounting. Encrypted snapshots go to
   `MyDrive/TextGenEncrypted`. A previous latest snapshot is copied into the
   runtime only if there is no active local vault file.
5. Run cell 4. Create/unlock the vault in the embedded browser interface, then
   select and load the GGUF on the **Model** tab. Passwords, prompts and responses
   must not be entered into notebook cells.
6. Finish with **Save and lock**, wait for success, then run the stop cell.

To bring your laptop's vault, download an **Encrypted backup** from its toolbar.
Upload that file to `MyDrive/TextGenEncrypted/vault-latest.sqlite3` before cell 3
in a fresh runtime. Unlock with the existing password. Do not copy a running
SQLite database while its WAL is active.

## Persistence

The active database is `/content/textgen-encrypted-vault.sqlite3`. It contains
ciphertext, including in its WAL. The live database is deliberately kept off
the Drive mount because SQLite locking/journaling requires filesystem behavior
that should not be assumed for a cloud-backed mount.

The gateway creates complete encrypted SQLite snapshots about once per minute
after changes, and on explicit backup, password change, unlock, and successful
save/lock. It writes and fsyncs a temporary snapshot before rotating
`vault-latest.sqlite3` to `vault-previous.sqlite3` and publishing the new latest
file. The toolbar shows the latest completed backup time. Google Drive may
still delay or interrupt remote synchronization; keep a downloaded backup for
important work. Older snapshots retain their original passwords.

Colab can end a runtime abruptly. Changes newer than the last completed Drive
snapshot can be lost. A backup failure stops the worker and retains RAM data
for retry. If the latest snapshot is unavailable, the previous snapshot can be
copied back as the latest in a fresh runtime, preserving both originals first.

## Isolation without Landlock

`--vault-isolation colab` requires the root notebook runtime, bubblewrap and
libseccomp2.
The supervisor reserves an unused host UID/GID for its worker and drops to that
identity **before** running bubblewrap. The worker gets a read-only view of the
runtime filesystem, a private RAM-backed `/dev`, selected NVIDIA character
devices, and access to its writable tmpfs workspace. Capabilities are dropped,
new privileges are disabled, and an inherited seccomp filter blocks new user
namespaces and mount changes. The filter is installed by bubblewrap before
Python executes; it does not need to write Colab's read-only `/proc/sys`.
Children inherit the mount restrictions. No disk-backed temporary directory
is substituted.

Colab denies creating new proc mounts. This launcher reuses the read-only proc
view, while the separate host identity prevents traversal through the notebook
or supervisor's `/proc/PID/root` and descriptors. Startup verifies the actual
reachable mount flags and capabilities, tests denial against a world-writable host file
and parent proc paths, and checks child-process inheritance. The worker repeats
the mount/privilege checks before importing TextGen or model libraries.

The supervisor owns the RAM workspace's outer directory (0710, traversal only
for the private worker group); data and volatile subdirectories are private to
the worker. This permits authenticated cleanup after a crash without leaving
an unowned workspace behind. Only the supervisor sees the vault encryption key.

This is not isolation from Google, a root process, or other notebook code with
root privileges. Plaintext must exist in RAM/VRAM during inference. GPU drivers,
the host kernel and cloud infrastructure remain trusted. The sandbox is for
preventing application filesystem writes from leaving plaintext on persistent
storage; it is not a claim of protection from every native-code vulnerability.

## Browser access

The gateway still binds only `127.0.0.1`. The notebook uses Colab's private port
proxy and `serve_kernel_port_as_iframe(..., cache_in_notebook=False)`. It does
not expose a public Gradio tunnel. The iframe opens the exact session HTTPS
origin configured with `--vault-colab-origin`. Its initial landing request uses
a random, one-use capability to pin the proxy's actual transport Host, which
may differ from the browser URL. A redirect removes the capability before
showing the unlock form. It never unlocks the vault or authorizes another host.
Other origins and unconfigured hosts are rejected. Forwarded
headers are generated by the gateway, secure HttpOnly partitioned cookies
protect the session, CSRF tokens protect controls, and plaintext responses use
`no-store`. Colab frame ancestors are allowed so the full UI can be embedded.

If the interface cannot retain its login, check the browser's permissions for
Colab's embedded frames. If the proxy format changes, the launcher stops rather
than trusting an arbitrary hostname. The session URL changes with the runtime.
The notebook stores no password and saves no rendered chat content in its cell
outputs. Avoid manually printing private data into notebook output.
Diagnostics opens within the authenticated frame so its partitioned cookie
remains available. The stop cell refuses to end a session with an unlocked or
retained RAM workspace; complete **Save and lock** first.

Google's [usage restrictions](https://research.google.com/colaboratory/faq.html#limitations-and-restrictions)
continue to apply, including the free-tier restriction on primarily using a
separate web interface. This code adds no keepalive, concealment, or resource
restriction bypass. Runtime availability and allowed usage remain Google's
decisions.

## Troubleshooting

- **Invalid Host in an older notebook:** use the updated cell 4 as well as the
  updated repository; pulling the repository does not replace cells in an
  already-open notebook. If no vault was unlocked, run cell 5 to stop the old
  gateway. In the same runtime, the following new cell updates the code and
  executes the current launch cell, keeping the existing model and Drive mount:

  ```python
  import json
  import subprocess

  subprocess.run(["git", "-C", str(TEXTGEN_ROOT), "pull", "--ff-only"], check=True)
  notebook = json.loads((TEXTGEN_ROOT / "Colab-Encrypted-TextGen.ipynb").read_text())
  launch = next("".join(cell["source"]) for cell in notebook["cells"]
                if "".join(cell["source"]).startswith("#@title 4."))
  exec(launch)
  ```

- **Isolation preflight failed:** do not remove the guard. The message reports
  whether bubblewrap, user namespaces, filesystem flags, or the privilege drop
  failed. No vault has been unlocked. This runtime cannot run this backend as
  configured; the local Landlock launcher remains available.
- **Model load failed:** open **Diagnostics** in the unlocked toolbar. Backend
  logs remain inside the encrypted vault; NVIDIA/GPU errors need investigation
  on the assigned runtime.
- **`MemoryError` in `metadata_gguf.py`, before llama.cpp starts:** the old reader
  trusted lengths in the file without validating its signature. An HTML file
  page saved with a `.gguf` extension reproduces this error. It does not by
  itself establish a GPU memory shortage. The updated reader checks the GGUF
  signature, version and metadata lengths and reports a useful file error.
  HTML pages, Git LFS pointers and malformed/truncated metadata are rejected.
  Supported files use little-endian GGUF v2/v3; metadata is limited to 128 MiB
  and 2,000,000 entries/array elements in total, with 65535-byte maximum keys.
  These limits apply to metadata, not model weight size.

  Cell 2 now converts Hugging Face `/blob/` file-page URLs into `/resolve/`
  download URLs. It checks new and existing files; a bad existing file is kept
  until a replacement download passes header/metadata validation. Interrupted
  downloads are not installed, and HTTP Content-Length is checked when supplied.
  Metadata checks do not verify tensor contents or the publisher's checksum.

  To repair an existing session, **Save and lock**, then run cell 5. In a new
  cell, use the same public model URL that you originally downloaded:

  ```python
  import json
  import subprocess

  model_url = ""  # Paste your public GGUF URL here, or leave blank to check existing files.
  subprocess.run(["git", "-C", str(TEXTGEN_ROOT), "pull", "--ff-only"], check=True)
  subprocess.run([str(TEXTGEN_PYTHON), "-B", "-m", "modules.gguf_download",
                  "--model-dir", str(TEXTGEN_MODELS), "--url", model_url],
                 cwd=TEXTGEN_ROOT, check=True)
  notebook = json.loads((TEXTGEN_ROOT / "Colab-Encrypted-TextGen.ipynb").read_text())
  launch = next("".join(cell["source"]) for cell in notebook["cells"]
                if "".join(cell["source"]).startswith("#@title 4."))
  exec(launch)
  ```

  This keeps the current vault, Drive mount and valid model downloads. Pulling
  code alone does not replace cells in an already-open notebook; the snippet
  calls the new downloader and launch cell explicitly. If validation still
  fails, share the error and the public model URL (remove any access tokens).
- **Drive backup failed:** keep the runtime alive, correct Drive access/quota,
  re-enter the password if requested, and retry **Save and lock**. Do not discard
  the active runtime while it contains unsaved RAM data.
- **Already running:** use the existing interface, save/lock, then run cell 5
  before rerunning the launch cell. Do not start a second gateway on the same
  vault or port.

Implementation references: [bubblewrap](https://github.com/containers/bubblewrap),
[Colab iframe API](https://github.com/googlecolab/colabtools/blob/main/google/colab/output/_util.py).
