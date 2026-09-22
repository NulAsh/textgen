# TextGen encrypted vault

This branch is based on upstream commit
`c93f8871239550de2ccfe1e95d469aa82616f07e`. It implements one password-protected
vault for one user, unlocked in a web browser. It does not implement multiple
user accounts or end-to-end encryption against the computer running inference.

## Install and start

Requirements:

- Linux x86_64 or aarch64 with Landlock ABI 3 or newer enabled.
- Python 3.11+ with SQLite serialization support (standard CPython builds).
- A writable tmpfs, normally `/dev/shm`, with room for the active user data.
- Your normal TextGen dependencies and an already downloaded model.
- Encrypted swap and hibernation storage, or no swap/hibernation, if you also
  need to protect operating-system memory images. tmpfs itself can be swapped.

For a new source installation, from the repository directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/portable/requirements_vulkan.txt
python server.py --portable --model-dir /absolute/path/to/models --auto-launch
```

Use the appropriate existing requirements file for your GPU/backend. For an
environment that already has TextGen installed, add the vault dependencies:

```bash
python -m pip install -r requirements/vault.txt
python server.py --portable --model-dir /absolute/path/to/models
```

Open `http://127.0.0.1:7860`. On first use, choose and confirm a long password
(minimum 12 characters). On subsequent starts, enter that password. The model
and chat interface start after unlocking. The password is never a CLI argument
or an environment variable, and it is not written to the vault.

Normal model flags, such as `--model`, `--gpu-layers`, `--ctx-size` and
`--cache-type`, are forwarded to TextGen. The gateway has its own `--help`.
External `CMD_FLAGS.txt` is not loaded by the launcher; pass flags explicitly.
Do not use the upstream Electron launcher or its prebuilt releases for this
branch. Use `python server.py` and the browser interface.

Optional gateway arguments:

| Argument | Purpose |
| --- | --- |
| `--vault /path/vault.sqlite3` | Choose the encrypted database. Default: `user_data/vault.sqlite3` beside the source tree. |
| `--listen-port 7860` | Change the local browser port. Binding remains `127.0.0.1`. |
| `--vault-ram-dir /dev/shm` | Select an existing tmpfs directory. A disk filesystem is refused. |
| `--vault-import /old/textgen/user_data` | Offer import when creating a new vault. |
| `--model-dir`, `--image-model-dir`, `--lora-dir` | Locations of existing model weights, opened read-only by the worker. |

There is no fallback that starts TextGen unencrypted or without write isolation.
An unsupported kernel/container fails before loading user content. A standard
container may hide the Landlock syscalls even when the host kernel supports them.

## Import existing data

Stop the original instance and start this fork with a new vault path:

```bash
python server.py --portable \
  --vault /absolute/path/private/vault.sqlite3 \
  --vault-import /absolute/path/old-textgen/user_data \
  --model-dir /absolute/path/old-textgen/user_data/models
```

On the create-vault screen, select the import checkbox. It copies chats,
notebooks, settings, characters, users, presets, templates, grammars, caches,
image outputs and training data into the encrypted vault. Model-specific
`models/config-user.yaml` becomes the encrypted `model-settings.yaml` file.
Model weights are not copied. Imported extensions, executable tools, MCP
configuration and command flags are not activated. Symlinks and hardlinks are
rejected rather than followed.

**The original files remain unchanged and unencrypted.** The checkbox explicitly
acknowledges this. Verify the imported content and take an encrypted backup
before deciding how to handle the original installation, exports and backups.
Deleting an original file is not a guarantee of erasure from SSDs, snapshots or
backups. This fork does not promise secure deletion.

## Saving, locking and backups

TextGen works inside a private mode-0700 tmpfs directory while unlocked. Its
regular files are copied into the encrypted database once per second when a
stable snapshot can be captured. Framework uploads, thumbnails and image
metadata follow this same storage path. Other temporary and library cache files
stay in a volatile tmpfs directory and are discarded on locking.

TextGen retains its own autosave timing. Therefore, a crash/power loss can lose
work since the last application save plus the last completed encrypted
checkpoint; this is not synchronous persistence on every keystroke. A file
changing during capture is retried without replacing the previous checkpoint.

Use **Save and lock** in the toolbar. This invalidates the browser session,
disconnects active proxy requests, stops generation and the worker, flushes
pending settings/notebook saves, saves a final encrypted snapshot, removes the
RAM tree and releases the key references. Locking while a backend is stalled
may require killing it; only content that reached the application's save files
can be preserved. Wait for generation to finish if you need its complete output.

If encryption or disk writes fail, the app stops accepting normal use and keeps
the RAM workspace. It does **not** report a successful lock or delete the last
copy of unsaved data. Correct the storage problem, re-enter the password and
retry **Save and lock**. Do not reboot if you need to recover unsaved RAM data.
The diagnostic view is authenticated and marked `no-store`.

**Encrypted backup** downloads a consistent, single-file SQLite backup. It
contains encrypted content only. Use this button instead of copying a live
database while its WAL is active. To restore, stop TextGen and launch with the
backup as a *new* `--vault` path; do not overwrite a running database. Keep an
untouched backup copy first.

**Change password** verifies the current password, generates a fresh random
data key and re-encrypts the saved records in one transaction. Historical
backups still use the original password. Forgotten passwords cannot be reset.

Plaintext chat/session export buttons direct you to the encrypted backup
control. Deliberate browser actions such as copying text, saving an image,
printing, developer tools or screenshots are outside the storage layer.

## Architecture and cryptography

1. The local aiohttp gateway serves the unlock page. No model or Gradio UI is
   running while the vault is locked.
2. Argon2id derives a 256-bit wrapping key from the password and a random
   128-bit salt: 64 MiB memory, 3 iterations, 4 lanes. The version-1 KDF
   parameters are fixed and validated before derivation.
3. AES-256-GCM unwraps a random 256-bit data key. HKDF-SHA256 separates record
   encryption and filename-index keys.
4. Each serialized path/content record has a fresh random 96-bit nonce and an
   authentication tag. Associated data binds it to the format, vault identity
   and opaque record ID. Filenames are encrypted; indexed IDs use HMAC-SHA256.
5. An encrypted manifest authenticates the inventory and record hashes. Missing,
   modified and substituted records fail integrity checks. SQLite transactions,
   WAL and full synchronization protect checkpoint atomicity. Journals and
   backups see ciphertext, never plaintext records or keys.
6. After successful unlock, the data is restored into tmpfs. A separate worker
   inherits Linux Landlock restrictions before importing TextGen/Gradio or
   creating model threads. Ordinary filesystem creation, writes, truncation,
   removal and renaming outside that RAM tree are denied. `/dev/null` remains
   writable. This is a file-write boundary, not a sandbox for malicious code.
7. Gradio binds a Unix socket inside the private directory. The gateway proxies
   authenticated HTTP and SSE requests to it. A random HttpOnly, SameSite=Strict
   session cookie, strict Host/Origin checks and CSRF tokens protect controls.
   Responses containing content are marked `no-store`. Access logs are disabled.
8. Worker/model stdout and stderr go to a file inside the vault, not the
   launcher console. Model-specific user settings are redirected from the
   external model folder into the vault. The llama.cpp service gets a random
   per-launch API key through its environment, with a matching client header.

The implementation uses `cryptography`'s high-level Argon2id/AESGCM/HKDF APIs.
It does not implement a cipher or password hash itself.

## Supported scope and limits

- **Covered persistence:** chats and titles, message versions, custom system
  messages, character/user definitions, notebook input and output, settings,
  uploaded files, profile images, generated images and embedded prompts,
  Gradio caches, and worker diagnostic logs.
- **Single user, local browser only.** Separate API serving, remote listening,
  public sharing, Electron, arbitrary extensions/custom tools/MCP, training and
  in-app model downloads are disabled. Model directories are read-only.
- Image generation saving is covered, but this change has not been exercised
  with a diffusion model. The same applies to GPU/native model execution and
  backend-specific JIT/multiprocessing features. Some libraries may need
  additional RAM-cache path integration; unexpected disk writes are refused.
- Default version-1 limits are 256 MiB per persistent file and 2 GiB total
  active file data. Capture/restore can temporarily require additional RAM.
  A limit failure stops operation and retains the working tree for recovery.
- The vault hides contents and filenames, not file count, ciphertext sizes,
  write timing or the existence of a vault. Replaying an entire older valid
  database is not detectable without an external trusted freshness record.
- Plaintext necessarily exists in the browser, Python, model RAM/VRAM and
  tmpfs during use. Swap, hibernation, browser profiles/extensions, OS crash
  capture, root access and processes running as the same OS user are outside
  the encryption boundary. Core dumps are disabled for the gateway and worker;
  Python cannot guarantee secure zeroization of all memory copies.
- `SIGKILL`, supervisor crashes and power failure cannot perform an orderly
  final save. A RAM tree may survive a supervisor crash until the next unlock
  or reboot. The worker is bound to its supervisor's lifetime. New unlocks
  restore the last encrypted checkpoint, not an unverified leftover RAM tree.
- Web searches, if enabled, send queries to external services. Encryption here
  concerns local persistence; it cannot govern third-party service storage.
- This is a reviewable first implementation, not an independently audited
  security product. Read the validation report before trusting it with data.

## Tests

Storage and gateway tests need only the vault dependencies plus pytest:

```bash
python -m pip install -r requirements/vault.txt pytest pytest-asyncio
python -m pytest -q tests/vault
```

The real-interface probe also needs TextGen's portable dependencies. The native
Landlock test runs when the current kernel exposes ABI 3+; otherwise it reports
a skip. There is no production option to disable that requirement. The proxy
test uses a loopback transport stand-in with synthetic data where Unix sockets
are unavailable. No test opens a real vault without its password.

See [validation details](Vault-Validation.md) for what was actually exercised.

## References

- [cryptography key derivation](https://cryptography.io/en/stable/hazmat/primitives/key-derivation-functions/)
- [cryptography authenticated encryption](https://cryptography.io/en/stable/hazmat/primitives/aead/)
- [Argon2 specification, RFC 9106](https://www.rfc-editor.org/rfc/rfc9106)
- [Linux Landlock documentation](https://docs.kernel.org/userspace-api/landlock.html)
