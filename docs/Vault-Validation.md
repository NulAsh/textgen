# Encrypted vault validation

Validated against upstream `c93f8871239550de2ccfe1e95d469aa82616f07e`.
All fixtures contain synthetic data. No user chats or passwords were used.

## Local results

The suite was run with Python 3.12.14, cryptography 50.0.1, the upstream portable
requirements and Gradio 4.37.2+custom.21:

```text
python -m pytest -q -rs tests/vault
39 passed, 2 skipped

ruff check modules/vault_*.py tests/vault --output-format concise
All checks passed!

git diff --check
No whitespace errors
```

The two native tests are skipped in the development container because it does
not expose the Landlock syscalls. It also prohibits Unix-domain sockets. The
production launcher refuses to run under those conditions; there is no bypass
flag. The CI workflow requires Landlock before running the suite on Linux.
See the repository's Actions results for the native run status.

## What the tests exercise

| Area | Validation |
| --- | --- |
| Content coverage | Real TextGen chat history, system settings, notebook input/output, model templates, Gradio upload and attachment paths, generated PNG prompt metadata |
| Round-trip integrity | Unicode filenames, binary uploads, directories, rename/deletion and exact restored contents |
| Ciphertext persistence | Synthetic content, filenames and password absent from database/WAL/backup bytes; distinct nonces across rewrites |
| Authentication | Wrong password rejection, password rotation and historical encrypted backup access |
| Tamper detection | Modified ciphertext, removed objects, substituted identifiers, missing/modified manifest, invalid KDF parameters |
| Failure handling | Simulated disk errors preserve the previous encrypted snapshot; failed lock retains unsaved RAM data and invalidates the session |
| HTTP controls | Locked route rejection, session revocation, Host/Origin/CSRF rejection, authenticated streaming and uploads, no-store responses |
| Filesystem input | Reject symlinks, hardlinks, special files, missing roots and disk-backed runtime directories |
| Shutdown | Pending edits flush; old completed notebook input is not replayed over newer generated output |
| Native tests | Child-process write isolation and a real confined TextGen worker using its private Unix socket; conditional on kernel support |

The real-interface probe constructs TextGen's Gradio interface in-process,
requests its actual `/config` and `/upload` ASGI routes and calls the upstream
save functions. It then encrypts and restores the resulting files and checks
their contents. It does not load a model or generate real inference output.

The gateway streaming test uses a loopback transport stand-in where the
development container prohibits Unix sockets. It never starts an unconfined
TextGen worker. The separate native-worker test uses the production launcher
and confinement without monkeypatching them.

## Still requiring deployment validation

- Real model inference, GPU/VRAM behavior, backend JIT caches and native
  subprocesses, including the llama.cpp authentication integration.
- Diffusion model execution and large image workloads.
- Browser interaction across the full Gradio interface, including model
  switching and long-running streamed generations.
- Abrupt power loss, full operating-system recovery, large vault performance,
  swap/hibernation configuration and deployment-specific containers.
- Independent cryptographic and application-security review.

Start with synthetic data on the intended machine and backend. Passing this
suite is evidence for the exercised paths, not a security audit or a guarantee
that every upstream feature works under confinement.
