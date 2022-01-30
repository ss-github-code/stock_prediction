from django.urls import path
from . import views

urlpatterns = [
    path('', views.homepage, name='homepage'),
    path('get_result', views.get_result, name='get_result'),
    path('report', views.report, name='report')
]
