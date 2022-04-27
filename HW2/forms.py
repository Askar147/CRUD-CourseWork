from django import forms
from django.db.models import fields
from .models import Diseasetype, Record

class DiseaseForm(forms.ModelForm):

    class Meta:
        model = Record
        fields = '__all__'

    def __init__(self, *args,**kwargs):
        super(DiseaseForm, self).__init__(*args,**kwargs)
        self.fields['email'].empty_label = "--Choose--"
        self.fields['cname'].empty_label = "--Choose--"
        self.fields['diseasecode'].empty_label = "--Choose--"
         