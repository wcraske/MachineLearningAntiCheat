// Copyright Epic Games, Inc. All Rights Reserved.
#include "FPSHUD.h"
#include "TelemetryFPS4Character.h"
#include "Engine/Canvas.h"
#include "Engine/Texture2D.h"
#include "Engine/Font.h"
#include "Kismet/GameplayStatics.h"
void AFPSHUD::DrawHUD()
{
	AmmoCountText = TEXT("Ammo: ");
	Super::DrawHUD();


	if (CrosshairTexture)
	{
		// Find the center of our canvas.
		FVector2D Center(Canvas->ClipX * 0.5f, Canvas->ClipY * 0.5f);

		// Offset by half of the texture's dimensions so that the center of the texture aligns with the center of the Canvas.
		FVector2D CrossHairDrawPosition(Center.X - (CrosshairTexture->GetSurfaceWidth() * 0.5f), Center.Y - (CrosshairTexture->GetSurfaceHeight() * 0.5f));

		// Draw the crosshair at the centerpoint.
		FCanvasTileItem TileItem(CrossHairDrawPosition, CrosshairTexture->Resource, FLinearColor::White);
		TileItem.BlendMode = SE_BLEND_Translucent;
		Canvas->DrawItem(TileItem);
	}

	ATelemetryFPS4Character* PlayerCharacter = Cast<ATelemetryFPS4Character>(UGameplayStatics::GetPlayerCharacter(GetWorld(), 0));


    if (PlayerCharacter)
    {
        // Position for ammo counter (bottom-right corner)
        FVector2D AmmoPosition(Canvas->ClipX * 0.75f, Canvas->ClipY * 0.75f);
        // Create the ammo text string
        FString AmmoText = FString::Printf(TEXT("Ammo: %d"), PlayerCharacter->GetAmmoCount());
		
		FCanvasTextItem TextItem(AmmoPosition, FText::FromString(AmmoText), GEngine->GetSmallFont(), FLinearColor::Red);	
		TextItem.Scale = FVector2D(2.0f, 2.0f);  
		Canvas->DrawItem(TextItem);


        
    }

}

