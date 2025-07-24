// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "TelemetryFPS4GameModeBase.generated.h"

/**
 * 
 */
UCLASS()
class TELEMETRYFPS4_API ATelemetryFPS4GameModeBase : public AGameModeBase
{
	GENERATED_BODY()

	virtual void StartPlay() override;
	
};

