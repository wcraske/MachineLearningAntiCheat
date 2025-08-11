// Fill out your copyright notice in the Description page of Project Settings.
//https://www.orfeasel.com/actor-spawning/

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Enemy.h"
#include "EnemySpawner.generated.h"

UCLASS()
class TELEMETRYFPS4_API AEnemySpawner : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	AEnemySpawner();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	//BP reference of enemy class
	UPROPERTY(EditDefaultsOnly, Category = "Spawning")
	TSubclassOf<AEnemy> EnemyBP;

	//delay after BP of enemy spawned
	
	UPROPERTY(EditDefaultsOnly, Category = "Spawning")
	float SpawnTime = 5.0f;

	//spawns enemy BP
	UFUNCTION()
	void SpawnEnemy();


};


