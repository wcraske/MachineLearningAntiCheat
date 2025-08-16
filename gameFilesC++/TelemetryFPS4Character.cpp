// Copyright Epic Games, Inc. All Rights Reserved.

#include "TelemetryFPS4Character.h"
#include "Camera/CameraComponent.h"
#include "GameFramework/InputSettings.h"
#include "FPSProjectile.h"
#include "DrawDebugHelpers.h"
#include "Kismet/GameplayStatics.h"
#include "Enemy.h"
#include "AntiCheatGuard.h"
#include "HttpModule.h"
#include "Interfaces/IHttpRequest.h"
#include "Interfaces/IHttpResponse.h"
#include "Http.h"


// Sets default values
ATelemetryFPS4Character::ATelemetryFPS4Character()
{
	// Set this character to call Tick() every frame. You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	// Create a first person camera component.
	FPSCameraComponent = CreateDefaultSubobject<UCameraComponent>(TEXT("FirstPersonCamera"));
	check(FPSCameraComponent != nullptr);

	// Attach the camera component to our capsule component.
	FPSCameraComponent->SetupAttachment(CastChecked<USceneComponent, UCapsuleComponent>(GetCapsuleComponent()));

	// Position the camera slightly above the eyes.
	FPSCameraComponent->SetRelativeLocation(FVector(0.0f, 0.0f, 50.0f + BaseEyeHeight));

	// Enable the pawn to control camera rotation.
	FPSCameraComponent->bUsePawnControlRotation = true;

	// Create a first person mesh component for the owning player.
	FPSMesh = CreateDefaultSubobject<USkeletalMeshComponent>(TEXT("FirstPersonMesh"));
	check(FPSMesh != nullptr);

	// Only the owning player sees this mesh.
	FPSMesh->SetOnlyOwnerSee(true);

	// Attach the FPS mesh to the FPS camera.
	FPSMesh->SetupAttachment(FPSCameraComponent);

	// Disable some environmental shadowing to preserve the illusion of having a single mesh.
	FPSMesh->bCastDynamicShadow = false;
	FPSMesh->CastShadow = false;

	// The owning player doesn't see the regular (third-person) body mesh.
	GetMesh()->SetOwnerNoSee(true);

	// Create and attach gun mesh
	GunMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("GunMesh"));
	GunMesh->SetupAttachment(FPSMesh, TEXT("GripPoint"));
	GunMesh->SetOnlyOwnerSee(true); // Only visible in first person
	GunMesh->bCastDynamicShadow = false;
	GunMesh->CastShadow = false;

	//default ofset for the muzzle location
	MuzzleOffset = FVector(100.0f, 0.0f, 0.0f); // Offset for the muzzle location

	//default firing mode
	bUseHitscan = true;

	
}

// Called when the game starts or when spawned
void ATelemetryFPS4Character::BeginPlay()
{
	Super::BeginPlay();


	if (GEngine)
	{
		// Display a debug message for five seconds.
		// The -1 "Key" value argument prevents the message from being updated or refreshed.
		GEngine->AddOnScreenDebugMessage(-1, 5.0f, FColor::Red, TEXT("We are using FPSCharacter."));
	}


}


// Called every frame
void ATelemetryFPS4Character::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	if (bPressedAimbot)
	{
		Aimbot();
	}

	float CurrentTime = GetWorld()->GetTimeSeconds();
	if (!bCurrentlyFiring && (CurrentTime - LastTelemetrySendTime > TelemetrySendInterval))
	{
		SendTelemetry();
		LastTelemetrySendTime = CurrentTime;
	}
}


// Called to bind functionality to input
void ATelemetryFPS4Character::SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent)
{
	Super::SetupPlayerInputComponent(PlayerInputComponent);

	// Set up "movement" bindings.
	PlayerInputComponent->BindAxis("MoveForward", this, &ATelemetryFPS4Character::MoveForward);
	PlayerInputComponent->BindAxis("MoveRight", this, &ATelemetryFPS4Character::MoveRight);

	// Set up "look" bindings.
	PlayerInputComponent->BindAxis("Turn", this, &ATelemetryFPS4Character::AddControllerYawInput);
	PlayerInputComponent->BindAxis("LookUp", this, &ATelemetryFPS4Character::AddControllerPitchInput);

	// Set up "action" bindings.
	PlayerInputComponent->BindAction("Jump", IE_Pressed, this, &ATelemetryFPS4Character::StartJump);
	PlayerInputComponent->BindAction("Jump", IE_Released, this, &ATelemetryFPS4Character::StopJump);
	PlayerInputComponent->BindAction("Fire", IE_Pressed, this, &ATelemetryFPS4Character::Fire);
	PlayerInputComponent->BindAction("Aimbot", IE_Pressed, this, &ATelemetryFPS4Character::ToggleAimbot);

}

void ATelemetryFPS4Character::MoveForward(float Value)
{
	// Find out which way is "forward" and record that the player wants to move that way.
	FVector Direction = FRotationMatrix(Controller->GetControlRotation()).GetScaledAxis(EAxis::X);
	AddMovementInput(Direction, Value);
}

void ATelemetryFPS4Character::MoveRight(float Value)
{
	// Find out which way is "right" and record that the player wants to move that way.
	FVector Direction = FRotationMatrix(Controller->GetControlRotation()).GetScaledAxis(EAxis::Y);
	AddMovementInput(Direction, Value);
}

void ATelemetryFPS4Character::StartJump()
{
	bPressedJump = true;
}

void ATelemetryFPS4Character::StopJump()
{
	bPressedJump = false;
}



void ATelemetryFPS4Character::ToggleAimbot()
{
	bPressedAimbot = !bPressedAimbot;
}


void ATelemetryFPS4Character::RandomFire()
{
	Fire();
	//allow new timer to be set
	bFireScheduled = false; 
	
}


void ATelemetryFPS4Character::Aimbot()
{
	//get reference to the world, for finding closest enemy blueprint actor
	UWorld* World = GetWorld();
	if (!World) {
		return;
	}

	AActor* ClosestEnemy = nullptr;
	
	//get player camera location and rotation
	FVector CameraLocation;
	FRotator CameraRotation;
	GetActorEyesViewPoint(CameraLocation, CameraRotation);

	//get all enemies in the level
	TArray<AActor*> FoundEnemies;
	UGameplayStatics::GetAllActorsOfClass(World, AEnemy::StaticClass(), FoundEnemies);

	for (AActor* Enemy : FoundEnemies)
	{
		if (Enemy)
		{
			//get distance to enemy
			float DistanceToEnemy = FVector::Dist(CameraLocation, Enemy->GetActorLocation());
			//if this is the first enemy or closer than the previous closest
			if (!ClosestEnemy || DistanceToEnemy < FVector::Dist(CameraLocation, ClosestEnemy->GetActorLocation()))
			{
				//assign this enemy as the closest
				ClosestEnemy = Enemy;
				//get the enemy's location
				FVector EnemyLocation = Enemy->GetActorLocation();
				//get firection of camera to enemy
				FVector DirectionToEnemy = EnemyLocation - CameraLocation;
				//convert the direction to a rotation
				FRotator AimRotation = DirectionToEnemy.Rotation();
				//set camera rotation to aim at the enemy
				GetController()->SetControlRotation(AimRotation);
				if (!bFireScheduled)
				{
					float delay = FMath::RandRange(0.51f, 0.25f);
					GetWorld()->GetTimerManager().SetTimer(TriggerAimbot, this, &ATelemetryFPS4Character::RandomFire, delay, false);
					bFireScheduled = true;
				}
				//random delay to fire after a random delay
				//Fire();



			}
		}
	}


}






// Function to sent telemetry data
void ATelemetryFPS4Character::SendTelemetry()
{
	FString PayloadOut;
	if (TelemetryQueue.Dequeue(PayloadOut))
	{
		TSharedRef<IHttpRequest, ESPMode::ThreadSafe> Request = FHttpModule::Get().CreateRequest();
		Request->SetURL("http://127.0.0.1:8000/telemetry");
		Request->SetVerb("POST");
		Request->SetHeader("Content-Type", "application/json");
		Request->SetContentAsString(PayloadOut);
		Request->ProcessRequest();
	}

}




void ATelemetryFPS4Character::Fire()
{
	bCurrentlyFiring = true;

	ammoCount -= 1;

	// Get reference to world
	UWorld* World = GetWorld();
	if (!World) return;


	TArray<AActor*> Guards;
	UGameplayStatics::GetAllActorsOfClass(World, AAntiCheatGuard::StaticClass(), Guards);

	AAntiCheatGuard* Guard = Cast<AAntiCheatGuard>(Guards[0]);
	if (Guard)
	{
		Guard->HandleAmmoCount(ammoCount);
	}
	


	if (ammoCount <= 0)
	{
		ammoCount = 30;
	}


	//get camera location and rotation, then use them for viewpoint
	FVector CameraLocation;
	FRotator CameraRotation;
	GetActorEyesViewPoint(CameraLocation, CameraRotation);

	//calculate muzzle location by offsetting slightly from camera
	FVector MuzzleLocation = CameraLocation + FTransform(CameraRotation).TransformVector(MuzzleOffset);
	FRotator MuzzleRotation = CameraRotation;

	//test to adjust the start point of the trace to be slightly in front of the camera
	//replace all instances below with CameraLocation to replace Adjusted Start
	FVector AdjustedStart = CameraLocation + CameraRotation.RotateVector(FVector(0.0f, 0.0f, 50.0f));


	//get reference to world, for spawning line traces get world

	//if using hitscan default on
	if (bUseHitscan)
	{
		//set the end point of the hitscan trace, 10k units forward of the camera
		FVector TraceEnd = AdjustedStart + (CameraRotation.Vector() * 10000.0f);
		//stores what we hit
		FHitResult Hit;

		//setup trace params
		FCollisionQueryParams Params;
		Params.AddIgnoredActor(this); //dont hit self
		//perform raycast from cam position to trace end
		//using visibility collision channel
		bool bHit = World->LineTraceSingleByChannel(
			Hit,             
			AdjustedStart,   
			TraceEnd,         
			ECC_Visibility,  
			Params           
		);

		//if hit smth
// If the line trace hit something
		if (bHit)
		{
			// Get the actor that was hit by the trace
			AActor* HitActor = Hit.GetActor();

			// Draw a green debug line from the camera to the impact point to visualize the hit
			DrawDebugLine(World, AdjustedStart, Hit.ImpactPoint, FColor::Green, false, 0.2f, 0, 1.0f);

			// Check if the hit actor exists and can take damage
			if (HitActor && HitActor->CanBeDamaged())
			{
				FString Payload = FString::Printf(TEXT("{\"player_id\": \"%s\", \"aim_offset_x\": %.2f, \"aim_offset_y\": %.2f, \"hit_x\": %.2f, \"hit_y\": %.2f, \"hit_z\": %.2f}"),*GetName(), CameraRotation.Pitch, CameraRotation.Yaw, Hit.ImpactPoint.X, Hit.ImpactPoint.Y, Hit.ImpactPoint.Z);
				TelemetryQueue.Enqueue(Payload);
				//SendTelemetry(CameraRotation.Pitch, CameraRotation.Yaw, Hit.ImpactPoint); commented out for queue instead
				UGameplayStatics::ApplyPointDamage(
					HitActor,
					20.0f,
					CameraRotation.Vector(),
					Hit,
					GetController(),
					this,
					nullptr
				);


				// Display a debug message on screen showing the name of the hit actor
				GEngine->AddOnScreenDebugMessage(-1, 1.5f, FColor::Yellow,
					FString::Printf(TEXT("Hit: %s"), *HitActor->GetName()));
			}
		}
		else
		{
			// If no hit occurred, draw a red debug line to show the full trace path (missed shot)
			DrawDebugLine(World, AdjustedStart, TraceEnd, FColor::Red, false, 1.0f, 0, 1.0f);
		}


	}
	bCurrentlyFiring = false; 
}
