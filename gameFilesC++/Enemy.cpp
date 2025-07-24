// Fill out your copyright notice in the Description page of Project Settings.


#include "Enemy.h"
#include "Kismet/GameplayStatics.h"  // For getting player reference
#include "GameFramework/PlayerController.h"
#include "Blueprint/AIBlueprintHelperLibrary.h"   // For SimpleMoveToActor
#include "AIController.h"
#include "GameFramework/Controller.h" // For damage instigator
#include "Materials/MaterialInstanceDynamic.h"
#include "TimerManager.h"


// Sets default values
AEnemy::AEnemy()
{
 	// Set this character to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
    // Set the skeletal mesh to block visibility traces
    GetMesh()->SetCollisionResponseToChannel(ECC_Visibility, ECR_Block);

}

// Called when the game starts or when spawned
void AEnemy::BeginPlay()
{
	Super::BeginPlay();
	//get reference to player pawn, index 0 is the first player
	PlayerPawn = UGameplayStatics::GetPlayerPawn(GetWorld(), 0);


	
	
}

// Called every frame
void AEnemy::Tick(float DeltaTime)
{
    Super::Tick(DeltaTime);

    //if player exists
    if (PlayerPawn)
    {
        // Try to get this enemy's AIController by casting its current controller
        AAIController* AIController = Cast<AAIController>(GetController());

        // If the cast succeeded and we have a valid AIController
        if (AIController)
        {
            // Move the enemy toward the player using MoveToActor
            // enemy stops once it's within X units of the player
            AIController->MoveToActor(PlayerPawn, 25.0f);
        }
        else
        {
            UE_LOG(LogTemp, Error, TEXT("No AIController found!"));
        }
    }

}


// Called to bind functionality to input
void AEnemy::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
	Super::SetupPlayerInputComponent(PlayerInputComponent);

}

// Called when this actor takes any damage (used by ApplyPointDamage in player shooting logic)
//takes in the damage amount(from player fire method), damage event instigator controller(struct for info on damage)
// and damage-causing actor (struct for info on the actor that caused the damage)
float AEnemy::TakeDamage(float DamageAmount, FDamageEvent const& DamageEvent,
    AController* EventInstigator, AActor* DamageCauser)
{
    // Subtract the incoming damage from the enemy's health
    EnemyHealth -= DamageAmount;

    // Log the damage and updated health
    UE_LOG(LogTemp, Warning, TEXT("Enemy took %f damage. Health now: %f"), DamageAmount, EnemyHealth);

    // If health is zero or below, destroy the enemy actor
    if (EnemyHealth <= 0.0f)
    {
        UE_LOG(LogTemp, Warning, TEXT("Enemy has been destroyed!"));
        Destroy(); // Removes the actor from the game world



    }

    // Return the actual damage dealt (Unreal requires this)
    return DamageAmount;
}