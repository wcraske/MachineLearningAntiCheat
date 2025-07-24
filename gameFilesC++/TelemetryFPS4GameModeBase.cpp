// Copyright Epic Games, Inc. All Rights Reserved.


#include "TelemetryFPS4GameModeBase.h"

void ATelemetryFPS4GameModeBase::StartPlay()
{
	// Initialize the game mode here
	Super::StartPlay();
	//display a debug mssage for five seconds
	// The -1 "Key" value argument prevents the message from being updated or refreshed.
	check(GEngine != nullptr);
	GEngine->AddOnScreenDebugMessage(-1, 5.0f, FColor::Blue, TEXT("hello world"));
	
}
