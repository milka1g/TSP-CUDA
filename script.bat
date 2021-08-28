@echo off
SETLOCAL EnableDelayedExpansion

set cluster=

for /f %%G in ('dir /b /a-d *.txt') do (
	set /p cluster= < %%G
	@echo !cluster!
	if !cluster! leq 15 (
		START /WAIT C:\Users\mn170387d\Documents\"Visual Studio 2019"\Projects\CudaDiplomski\x64\Release\CudaDiplomski.exe %%G
	)
	if !cluster!==16 (
		START /WAIT C:\Users\mn170387d\Documents\"Visual Studio 2019"\Projects\CudaDiplomski14\x64\Release\CudaDiplomski14.exe %%G
	)
	if !cluster!==17 (
		START /WAIT C:\Users\mn170387d\Documents\"Visual Studio 2019"\Projects\CudaDiplomski15\x64\Release\CudaDiplomski15.exe %%G
	)

)


