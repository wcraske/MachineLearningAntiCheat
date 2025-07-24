// Fill out your copyright notice in the Description page of Project Settings.
#include "EnemySpawner.h"

// Sets default values
AEnemySpawner::AEnemySpawner()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;


}

// Called when the game starts or when spawned
void AEnemySpawner::BeginPlay()
{
	Super::BeginPlay();

	
	// used to control the timer
	FTimerHandle OutHandle;

	//gets world timer manager and sets a timer for the function SpawnEnemy
	//parameters: OutHandle, this (the object that owns the function, the spawner class), pointer to the member function(spawnEnemy), and the time delay, and looping set to trues
	GetWorld()->GetTimerManager().SetTimer(OutHandle, this, &AEnemySpawner::SpawnEnemy, SpawnTime, true);

}

// Called every frame
void AEnemySpawner::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

void AEnemySpawner::SpawnEnemy()
{
	//if EnemyBP is valid
	if (EnemyBP) 
	{
		//spawn parametres for current spawn
		FActorSpawnParameters SpawnParams;
		//used for random spawn location around spawner

		FVector NewSpawnLocation = GetActorLocation() + FVector(FMath::FRandRange(-200, 200),FMath::FRandRange(-200, 200),0);;
		FRotator NewSpawnRotation = GetActorRotation();
		FTransform NewSpawnTransform(NewSpawnRotation, NewSpawnLocation);

		//actual spawn, function returns reference to spawned actor
		AEnemy* ActorRef = GetWorld()->SpawnActor<AEnemy>(EnemyBP, GetTransform(), SpawnParams);
		GEngine->AddOnScreenDebugMessage(-1, 1.5f, FColor::Yellow,
			FString::Printf(TEXT("spawned enemy")));

	}
}




