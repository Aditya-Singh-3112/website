from django.urls import path
from .views import txt_to_img

urlspatterns = [
    path('text_to_img/', txt_to_img),
]