//https://learn.microsoft.com/en-us/windows/win32/psapi/enumerating-all-processes

#include "AntiCheatGuard.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <tchar.h>
#include <psapi.h>

AAntiCheatGuard::AAntiCheatGuard()
{
	PrimaryActorTick.bCanEverTick = true;
	bCheatDetected = false;
	checkIdx = 5;
	PreviousAmmo = -1;

	CheatProcesses = {
		TEXT("cheatengine.exe"),
		TEXT("cheatengine-x86_64.exe"),
		TEXT("cheatengine-i386.exe"),
		TEXT("cheatengine-x86_64-SSE4-AVX2.exe"),
		TEXT("OverwolfBrowser.exe"),
		TEXT("chrome.exe")
	};
}

void AAntiCheatGuard::BeginPlay()
{
	Super::BeginPlay();

	SetAmmoShadowCount(30);  // initial ammo
	PreviousAmmo = 30;

	ScanProcesses();

	if (bCheatDetected)
	{
		UE_LOG(LogTemp, Error, TEXT("Cheat detected!"));
		GEngine->AddOnScreenDebugMessage(-1, 5.0f, FColor::Yellow,
			TEXT("cheat detected"));
	}
}

void AAntiCheatGuard::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
}

void AAntiCheatGuard::ScanProcesses()
{
	DWORD aProcesses[1024], cbNeeded, cProcesses;

	if (!EnumProcesses(aProcesses, sizeof(aProcesses), &cbNeeded))
	{
		UE_LOG(LogTemp, Error, TEXT("Failed to enumerate processes"));
		return;
	}

	cProcesses = cbNeeded / sizeof(DWORD);

	for (unsigned int i = 0; i < cProcesses; i++)
	{
		if (aProcesses[i] != 0)
		{
			CompareProcessName(aProcesses[i]);
		}
	}
}

void AAntiCheatGuard::CompareProcessName(DWORD ProcessID)
{
	TCHAR szProcessName[MAX_PATH] = TEXT("<unknown>");
	HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, ProcessID);

	if (hProcess)
	{
		HMODULE hMod;
		DWORD cbNeeded;

		if (EnumProcessModules(hProcess, &hMod, sizeof(hMod), &cbNeeded))
		{
			GetModuleBaseName(hProcess, hMod, szProcessName, sizeof(szProcessName) / sizeof(TCHAR));
		}
		CloseHandle(hProcess);
	}

	FString ProcessName = FString(szProcessName);

	for (const FString& Cheat : CheatProcesses)
	{
		if (ProcessName.Equals(Cheat, ESearchCase::IgnoreCase))
		{
			bCheatDetected = true;
			UE_LOG(LogTemp, Warning, TEXT("Detected cheat process: %s"), *ProcessName);
			GEngine->AddOnScreenDebugMessage(-1, 1.5f, FColor::Yellow,
				FString::Printf(TEXT("cheat detected: %s"), *ProcessName));
			break;
		}
	}
}

// --- AMMO OBFUSCATION LOGIC ---

void AAntiCheatGuard::SetAmmoShadowCount(int32 ammo)
{
	ammoEncrypted = ammo ^ xorKey;
	ammoChecksum = ~(ammoEncrypted)+0xA5A5;
}

int32 AAntiCheatGuard::GetAmmoShadowCount()
{
	int32 expectedChecksum = ~(ammoEncrypted)+0xA5A5;

	if (expectedChecksum != ammoChecksum)
	{
		GEngine->AddOnScreenDebugMessage(-1, 5.0f, FColor::Red,
			TEXT("Ammo checksum mismatch: memory tampering detected"));
	}

	return ammoEncrypted ^ xorKey;
}

// --- AMMO MONITORING LOGIC ---

void AAntiCheatGuard::HandleAmmoCount(int32 ammo)
{
	if (PreviousAmmo != -1 && ammo < PreviousAmmo)
	{
		// Real ammo decreased; update shadow
		int32 diff = PreviousAmmo - ammo;
		int32 currentShadowAmmo = FMath::Max(0, GetAmmoShadowCount() - diff);
		SetAmmoShadowCount(currentShadowAmmo);
	}
	// Optional: handle increase (reloads, cheats, etc.)
	else if (PreviousAmmo != -1 && ammo > PreviousAmmo)
	{
		int32 diff = ammo - PreviousAmmo;
		int32 currentShadowAmmo = GetAmmoShadowCount() + diff;
		SetAmmoShadowCount(currentShadowAmmo);
	}

	PreviousAmmo = ammo;

	checkIdx -= 1;

	if (checkIdx <= 0)
	{
		CompareAmmoCount(ammo);
		checkIdx = 5;
	}
}

void AAntiCheatGuard::CompareAmmoCount(int32 ammo)
{
	int32 shadow = GetAmmoShadowCount();

	if (shadow != ammo)
	{
		GEngine->AddOnScreenDebugMessage(-1, 5.0f, FColor::Yellow,
			TEXT("Memory manipulated, ammo count does not match"));
	}
}
