# Generated by Django 3.2.9 on 2021-11-16 18:13

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Country',
            fields=[
                ('cname', models.CharField(max_length=50, primary_key=True, serialize=False)),
                ('population', models.BigIntegerField(blank=True, null=True)),
            ],
            options={
                'db_table': 'country',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='Diseasetype',
            fields=[
                ('id', models.IntegerField(primary_key=True, serialize=False)),
                ('description', models.CharField(blank=True, max_length=140, null=True)),
            ],
            options={
                'db_table': 'diseasetype',
                'managed': False,
            },
        ),
    ]