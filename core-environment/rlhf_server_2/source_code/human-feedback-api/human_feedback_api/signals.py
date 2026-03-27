import os
import pandas as pd
from django.db.models.signals import post_migrate
from django.dispatch import receiver
from django.conf import settings
from .models import Clip

@receiver(post_migrate)
def create_initial_clips(sender, **kwargs):
    if sender.name != 'human_feedback_api':
        return

    if Clip.objects.exists():
        return

    csv_path = os.path.join(settings.BASE_DIR, "clips.csv")
    if not os.path.exists(csv_path):
        print(f"No se encontró el archivo: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
        ip = settings.VIDEO_SERVER_HOST

        clips = []
        for _, row in df.iterrows():
            media_url = row['media_url'].replace("158.42.185.67", ip)

            clip = Clip(
                created_at= row['created_at'], 
                updated_at= row['updated_at'],
                media_url=media_url,
                environment_id=row['environment_id'],
                clip_tracking_id=row['clip_tracking_id'],
                domain=row['domain'],
                source=row['source'],
                actions=row['actions'] if pd.notna(row['actions']) else "",
            )
            clips.append(clip)

        Clip.objects.bulk_create(clips)
        print(f"{len(clips)} clips importados desde clips.csv")

    except Exception as e:
        print(f"Error al importar clips desde CSV: {e}")
