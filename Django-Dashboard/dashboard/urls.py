from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('classify/', views.classify, name='classify'),
    path("chat/", views.chat_api, name="chat_api"),
]
