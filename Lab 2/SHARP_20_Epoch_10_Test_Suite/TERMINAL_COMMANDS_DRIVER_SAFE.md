# One-command driver-safe start or resume

After updating the repository, run this in the Lab 2 foreground terminal:

```bash
set +e
set +u
set +o pipefail 2>/dev/null

PACKAGE="/home/server00/M/Codes/Thesis/Lab 2/SHARP_20_Epoch_10_Test_Suite"
cd "$PACKAGE"
bash "$PACKAGE/launch_lab2_10test_suite.sh"
STATUS=$?

echo
echo "Suite command status: $STATUS"
echo "Terminal remains open."
```

The launcher selects the verified non-root NVIDIA 580.159.03 user-space runtime when the loaded Lab 2 kernel module is still 580.159.03. It creates the experiment only once. On later invocations it reuses the pointer and resumes the incomplete variant.
