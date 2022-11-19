from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path("perform_login", views.perform_login, name="perform_login"),
    path("perform_logout", views.perform_logout, name="perform_logout"),
    path("admin_dashboard", views.admin_dashboard, name="admin_dashboard"),
    path("minhas_analises", views.minhas_analises, name="minhas_analises"),
    path("meus_documentos", views.meus_documentos, name="meus_documentos"),
    path("main", views.main, name="main"),
]

