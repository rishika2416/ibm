from django.urls import path
from . import views

urlpatterns = [
    path('', views.home1, name='home1'),
    path('home/', views.home, name='home'),
    path('oldtweets/', views.oldtweets, name='oldtweets'),
    path('news/', views.news, name='news'),
    path('statistic/', views.statistic, name='statistic'),
]