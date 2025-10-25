# Your Awesome Commands


Add here your awesome run commands for save them.

Remember the commands for seeing python tasks / cancel them: <br>
Linux:
```bash
ps aux | grep '[p]ython'
kill -9 -f python
```
Windows CMD:
```bash
tasklist /FI "IMAGENAME eq python.exe"
taskkill /IM python.exe /F
```
Windows PowerShell:
```bash
Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -like "*python*" } | Select-Object ProcessId, CommandLine
Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -like "*python*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

---

```bash
*add your run command here
```



