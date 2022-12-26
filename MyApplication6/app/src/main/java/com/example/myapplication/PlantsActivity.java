package com.example.myapplication;

import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;
import androidx.recyclerview.widget.DefaultItemAnimator;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.database.Cursor;
import android.os.Bundle;
import android.view.View;
import android.widget.Toast;

import java.util.ArrayList;

public class PlantsActivity extends AppCompatActivity {
    RecyclerView recyclerView;
    ArrayList<PlantModel> plantModels;
    DatabaseHelper databaseHelper;
    private RecyclerContactAdapter.RecyclerViewClickListener listener;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_plants);

        getSupportActionBar().hide();

        databaseHelper = new DatabaseHelper(this);

        recyclerView = findViewById(R.id.recyclerContact);
        plantModels = new ArrayList<>();

        getPlantInfo();
        setAdapter();
    }

    private void getPlantInfo() {
        Cursor cursor = databaseHelper.getData();
        if(cursor.getCount()==0){
            Toast.makeText(this,"No Entry Exists",Toast.LENGTH_SHORT).show();
            return;
        }else{
            while (cursor.moveToNext()){
                @SuppressLint("Range") PlantModel plantModel = new PlantModel(
                        cursor.getString(getResources().getIdentifier(String.valueOf(cursor.getColumnIndex("plantImg")),"drawable",getPackageName())),
                        cursor.getInt(cursor.getColumnIndex("id")),
                        cursor.getString(cursor.getColumnIndex("plantName")),
                        cursor.getString(cursor.getColumnIndex("plantStatement")),
                        cursor.getString(cursor.getColumnIndex("plantLink")));

                plantModels.add(plantModel);
            }
        }
    }
//cursor.getColumnIndex("plantImg")
    private void setAdapter() {
        setOnClickListener();
        RecyclerContactAdapter adapter = new RecyclerContactAdapter(plantModels,listener,this);
        recyclerView.setLayoutManager(new LinearLayoutManager(getApplicationContext()));
        recyclerView.setItemAnimator(new DefaultItemAnimator());
        recyclerView.setAdapter(adapter);
    }

    private void setOnClickListener() {
        listener = new RecyclerContactAdapter.RecyclerViewClickListener() {
            @Override
            public void onClick(View v, int position) {
                Intent intent = new Intent(getApplicationContext(),PlantActivity.class);
                intent.putExtra("result",position);
                startActivity(intent);
            }
        };
    }
}