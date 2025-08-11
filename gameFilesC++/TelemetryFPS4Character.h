	
// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "Camera/CameraComponent.h"
#include "Components/CapsuleComponent.h"
#include "Containers/Queue.h"
#include "FPSProjectile.h"
#include "Enemy.h"
#include "AntiCheatGuard.h"
#include "TelemetryFPS4Character.generated.h"

//forward declarations
class UAntiCheatManager;



UCLASS()
class TELEMETRYFPS4_API ATelemetryFPS4Character: public ACharacter
{
	GENERATED_BODY()

public:
	// Sets default values for this character's properties
	ATelemetryFPS4Character();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

	// Projectile class to spawn.
	UPROPERTY(EditDefaultsOnly, Category = Projectile)
	TSubclassOf<class AFPSProjectile> ProjectileClass;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
	int32 ammoCount = 30;

public:
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	
	// Called to bind functionality to input
	virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;

	// Handles input for moving forward and backward.
	UFUNCTION()
	void MoveForward(float Value);

	// Handles input for moving right and left.
	UFUNCTION()
	void MoveRight(float Value);

	// Sets jump flag when key is pressed.
	UFUNCTION()
	void StartJump();

	// Clears jump flag when key is released.
	UFUNCTION()
	void StopJump();

	// Function that fires projectiles.
	UFUNCTION()
	void Fire();
	bool bCurrentlyFiring;

	UFUNCTION()
	void ToggleAimbot();
	bool bPressedAimbot;
	


	// Function that starts aimbot.
	UFUNCTION()
	void Aimbot();


	// Function that sends telemetry.
	UFUNCTION()
	void SendTelemetry();
	//void SendTelemetry(float AimOffsetX, float AimOffsetY, FVector HitPoint);

	// FPS camera
	UPROPERTY(VisibleAnywhere)
	UCameraComponent* FPSCameraComponent;

	// First-person mesh (arms), visible only to the owning player.
	UPROPERTY(VisibleDefaultsOnly, Category = Mesh)
	USkeletalMeshComponent* FPSMesh;

	// Gun muzzle offset from the camera location.
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Gameplay)
	FVector MuzzleOffset;

	// Gun mesh: 1st person view (hidden in 3rd person)
	UPROPERTY(VisibleDefaultsOnly, Category = Mesh)
	UStaticMeshComponent* GunMesh;

	// Whether to use hitscan instead of physical projectile
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Gameplay)
	bool bUseHitscan;


	FTimerHandle TelemetryTimerHandle;

	UFUNCTION()
	void RandomFire();

	bool bFireScheduled = false;
	FTimerHandle TriggerAimbot;

	TQueue<FString> TelemetryQueue;

	float LastTelemetrySendTime = 0.0f;
	float TelemetrySendInterval = 0.2f;



	UFUNCTION()
	int32 GetAmmoCount() const { return ammoCount; }
	



};