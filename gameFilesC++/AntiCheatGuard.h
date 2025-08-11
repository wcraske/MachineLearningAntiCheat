#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "AntiCheatGuard.generated.h"

UCLASS()
class TELEMETRYFPS4_API AAntiCheatGuard : public AActor
{
	GENERATED_BODY()

public:
	AAntiCheatGuard();
	void CompareAmmoCount(int32 ammo);
	void HandleAmmoCount(int32 ammo);

protected:
	virtual void BeginPlay() override;
	virtual void Tick(float DeltaTime) override;

private:
	TArray<FString> CheatProcesses;
	bool bCheatDetected;

	void ScanProcesses();
	void CompareProcessName(DWORD ProcessID);

	// Anti-tamper ammo system
	int32 ammoEncrypted = 0;
	int32 ammoChecksum = 0;
	int32 xorKey = 0x5A3C2F1B;
	int32 checkIdx = 5;
	int32 PreviousAmmo;



	// Helper methods
	void SetAmmoShadowCount(int32 ammo);
	int32 GetAmmoShadowCount();
};
