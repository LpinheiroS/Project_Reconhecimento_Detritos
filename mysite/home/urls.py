from django.urls import path
from django.contrib import admin

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('admin_django', admin.site.urls, name='admin_django'),
    path("perform_login", views.perform_login, name="perform_login"),
    path("perform_logout", views.perform_logout, name="perform_logout"),
    path("admin_dashboard", views.admin_dashboard, name="admin_dashboard"),
    path("minhas_analises", views.minhas_analises, name="minhas_analises"),
    path("meus_documentos", views.meus_documentos, name="meus_documentos"),
    path("main", views.main, name="main"),
    path("cadastro_usuario", views.cadastro_usuario, name="cadastro_usuario"),
    path("cadastrar_usuario", views.cadastrar_usuario, name="cadastrar_usuario")
]

