from django.apps import AppConfig

class HumanFeedbackApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'human_feedback_api'

    def ready(self):
        import human_feedback_api.signals 
