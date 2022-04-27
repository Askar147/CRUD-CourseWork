from django.db.models import fields
import django_filters
from django_filters import NumberFilter

from .models import *

class RecordFilter(django_filters.FilterSet):

    from_patients = NumberFilter(field_name="totalpatients", lookup_expr='gte')
    to_patients = NumberFilter(field_name="totalpatients", lookup_expr='lte') 

    class Meta:
        model = Record
        fields = '__all__'
        exclude = ['totaldeaths', 'totalpatients', 'record_id']
    
