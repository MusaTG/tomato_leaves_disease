package com.example.myapplication;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.material.shape.CornerFamily;
import com.google.android.material.shape.MaterialShapeDrawable;

public class PlantActivity extends AppCompatActivity {

    Button videoBtn;
    TextView detailText, barTitle;
    DatabaseHelper databaseHelper;
    PlantModel plantModel;
    Integer result = 0;
    ImageView imageView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_plant);

        getSupportActionBar().hide();


        databaseHelper = new DatabaseHelper(this);

        videoBtn = findViewById(R.id.videoButton);
        detailText = findViewById(R.id.detailText);
        barTitle = findViewById(R.id.barTitle);
        imageView = findViewById(R.id.plantImg);

        Bundle extras = getIntent().getExtras();
        if (extras!=null){
            result = extras.getInt("result");
        }

        getPlantInfo();
        detailText.setText(plantModel.statement);
        barTitle.setText(plantModel.name);
        imageView.setImageResource(getResources().getIdentifier(plantModel.img,"drawable",getPackageName()));

        videoBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                goLink(plantModel.link);
            }
        });
    }

    @SuppressLint("Range")
    private void getPlantInfo() {
        Cursor cursor = databaseHelper.getDataId(result);
        if(cursor.getCount()==0){
            Toast.makeText(this,"No Entry Exists",Toast.LENGTH_SHORT).show();
            return;
        }else{
            while (cursor.moveToNext()){
                plantModel = new PlantModel(
                        cursor.getString(cursor.getColumnIndex("plantImg")),
                        cursor.getInt(cursor.getColumnIndex("id")),
                        cursor.getString(cursor.getColumnIndex("plantName")),
                        cursor.getString(cursor.getColumnIndex("plantStatement")),
                        cursor.getString(cursor.getColumnIndex("plantLink")));
            }
        }
    }

    private void goLink(String s) {
        Uri uri = Uri.parse(s);
        startActivity(new Intent(Intent.ACTION_VIEW,uri));
    }
}