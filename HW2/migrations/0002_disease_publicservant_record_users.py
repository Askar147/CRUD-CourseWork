# Generated by Django 3.2.9 on 2021-11-17 06:25

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('HW2', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Disease',
            fields=[
                ('diseasecode', models.CharField(max_length=50, primary_key=True, serialize=False)),
                ('pathogen', models.CharField(blank=True, max_length=20, null=True)),
                ('description', models.CharField(blank=True, max_length=140, null=True)),
            ],
            options={
                'db_table': 'disease',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='Users',
            fields=[
                ('email', models.CharField(max_length=60, primary_key=True, serialize=False)),
                ('name', models.CharField(blank=True, max_length=30, null=True)),
                ('surname', models.CharField(blank=True, max_length=40, null=True)),
                ('salary', models.IntegerField(blank=True, null=True)),
                ('phone', models.CharField(blank=True, max_length=20, null=True)),
            ],
            options={
                'db_table': 'users',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='Publicservant',
            fields=[
                ('email', models.OneToOneField(db_column='email', on_delete=django.db.models.deletion.DO_NOTHING, primary_key=True, serialize=False, to='HW2.users')),
                ('department', models.CharField(blank=True, max_length=50, null=True)),
            ],
            options={
                'db_table': 'publicservant',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='Record',
            fields=[
                ('email', models.OneToOneField(db_column='email', on_delete=django.db.models.deletion.DO_NOTHING, primary_key=True, serialize=False, to='HW2.publicservant')),
                ('totaldeaths', models.IntegerField(blank=True, null=True)),
                ('totalpatients', models.IntegerField(blank=True, null=True)),
                ('record_id', models.IntegerField(unique=True)),
            ],
            options={
                'db_table': 'record',
                'managed': False,
            },
        ),
    ]
