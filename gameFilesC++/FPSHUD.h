// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/HUD.h"
#include "Engine/Canvas.h"
#include "TelemetryFPS4Character.h"
#include "FPSHUD.generated.h"

/**
*
*/
UCLASS()
class TELEMETRYFPS4_API AFPSHUD : public AHUD
{
	GENERATED_BODY()

public:
	// Primary draw call for the HUD.
	virtual void DrawHUD() override;

protected:
	// This will be drawn at the center of the screen.
	UPROPERTY(EditDefaultsOnly)
	UTexture2D* CrosshairTexture;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite, Category = UI)
	FString AmmoCountText;

	// Font for UI text
	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite, Category = UI)
	class UFont* HUDFont;

	
};