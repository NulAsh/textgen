# Encrypted vault validation

Validated against upstream `c93f8871239550de2ccfe1e95d469aa82616f07e`.
All fixtures contain synthetic data. No user chats or passwords were used.

## Native model loader regression, 2026-11-12

A pristine llama_cpp_binaries 0.138.0 wheel contains versioned shared libraries
and a `_symlinks.json` manifest. Its import hook attempts to create the missing
SONAME aliases in the installed package. Landlock denies these writes, leaving
`llama-server` unable to find `libllama-common.so.0` (exit code 127). The earlier
interface-only native test did not load a model and missed this path.

The loader now resolves the manifest into symlinks in the worker's volatile RAM
directory and supplies that directory to the model subprocess's dynamic loader.
The executable and library files stay in the read-only installation. Regression
tests check chained aliases, repeated loads, invalid path/cycle rejection, and
an unchanged installation. The native probe imports a freshly extracted wheel
only after applying production Landlock, then loads the public tiny
`ggml-org/tiny-llamas` stories260K model and generates streamed tokens through
TextGen's `LlamaServer`. It also verifies HTTP 401 for an unauthenticated
generation request. CI downloads hash-pinned wheel/model fixtures for both
Python versions. No production confinement bypass was added.

Additional regressions clear the model dropdown through each empty value and
disconnect a real streaming HTTP client, checking that subsequent requests work
without a spurious gateway traceback. Existing tests retain coverage for genuine
proxy errors in encrypted diagnostics.

Local Python 3.12.14 and 3.14.10 tests each passed **49 tests with 3
Landlock-dependent skips**.
The CPU wheel and Vulkan wheel both loaded the tiny model and generated tokens
on the CPU. This does not exercise GPU offload.

[CI run 43174028306](https://github.com/NulAsh/textgen/actions/runs/43174028306)
passed **all 52 tests with no skips on both Python 3.11 and 3.14**, with Landlock
ABI 8. This includes fresh-wheel CPU model loading, streamed inference and
authentication under the production write restrictions, plus the real confined
web worker and queue. Ruff also passed in both jobs. This validates
implementation commit `d22bc665995c0efabbb22a60cbc4c65e2d767b37`.

## Queue regression, 2026-11-11

The initial validation did not submit a real Gradio queue job. A later report
identified HTTP 500 from `/ui/queue/join` despite successful page loading and
vault unlock. This was reproduced by constructing Gradio without a current
event loop: its queue locks were `None`. Python 3.14 uses that startup behavior.

The fix initializes Gradio's queue locks, app lock and stop event during ASGI
startup on the serving event loop. Added tests cover two successive queued
generator calls through `/queue/join` and `/queue/data`, including streamed and
completed output. The native worker test also submits TextGen's real page-load
event through the authenticated gateway and waits for successful completion.

Additional tests verify that proxy exception details are accessible only
through authenticated diagnostics and encrypted backups, and that password
request exceptions are excluded from detailed logging. CI now runs Python 3.11
and 3.14 with NumPy 2.3 and Pydantic 2.14.3. The historical results below apply
to the first implementation, before these regression tests were added.

Local runs on Python 3.12.14 and 3.14.10 each completed with **42 passed and
2 skipped** (the native Landlock tests). The queue regression failed with HTTP
500 before the fix and passed afterward on both versions. Ruff also passed.

[CI run 42982187397](https://github.com/NulAsh/textgen/actions/runs/42982187397)
passed **all 44 tests with no skips on both Python 3.11 and 3.14**, including
the production confined worker and its real queued page-load event. Both jobs
also passed Ruff. This validates implementation commit
`0a9c4aa1be06dbbc60177d939f40f386ea3ab48c`.

## Linux CI result

[Run 35703631813](https://github.com/NulAsh/textgen/actions/runs/35703631813)
passed on 2026-09-22 for implementation commit
`a6dae1d38b2173636350307a3db752b67b94b142`:

```text
Ubuntu 24.04.5, CPython 3.11.16, Landlock ABI 7
41 passed in 12.95s; no skipped tests
Ruff: All checks passed!
```

This includes the production confined worker, private Unix socket, real Gradio
configuration/upload routes, encrypted save, shutdown, RAM cleanup and restored
upload contents. The native subprocess test also verifies that writes outside
the allowed RAM directory are denied in both parent and child processes.
No inference model was loaded during this run.

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

- Production-sized model inference, GPU/VRAM behavior and backend JIT caches.
  The November 12 regression covers tiny-model CPU inference and llama.cpp
  authentication; other native backends still require deployment validation.
- Diffusion model execution and large image workloads.
- Browser interaction across the full Gradio interface, including model
  switching and long-running streamed generations.
- Abrupt power loss, full operating-system recovery, large vault performance,
  swap/hibernation configuration and deployment-specific containers.
- Independent cryptographic and application-security review.

Start with synthetic data on the intended machine and backend. Passing this
suite is evidence for the exercised paths, not a security audit or a guarantee
that every upstream feature works under confinement.
