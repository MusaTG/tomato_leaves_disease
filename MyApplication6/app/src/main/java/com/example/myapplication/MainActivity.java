package com.example.myapplication;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.myapplication.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    Button selectBtn, predictBtn, captureBtn, plantPageBtn, resultBtn;
    TextView resultText;
    ImageView imageView;
    Bitmap bitmap;
    DatabaseHelper databaseHelper;
    ArrayList<PlantModel> plantModels;
    int imageSize = 128;

    String[] labels = {
            "Bakteri Lekeli",
            "Erken Yanıklık",
            "Geç Yanıklık",
            "Yaprak Kalıbı",
            "Septoria Yaprak Lekesi",
            "Örümcek Akarları-İki Benekli",
            "Hedef Nokta",
            "Sarı Yaprak Kıvrılma Virüsü",
            "Mozaik Virüsü",
            "Sağlıklı"
    };

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getSupportActionBar().hide();

        databaseHelper = new DatabaseHelper(this);
        plantModels = new ArrayList<>();

        getPermission();

        selectBtn = findViewById(R.id.selectImage);
        captureBtn = findViewById(R.id.capture);
        imageView = findViewById(R.id.imageView);
        plantPageBtn = findViewById(R.id.plantPage);
        resultText = findViewById(R.id.resultText);
        predictBtn = findViewById(R.id.predict);
        resultBtn = findViewById(R.id.resultBtn);

        selectBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent();
                intent.setAction((Intent.ACTION_GET_CONTENT));
                intent.setType("image/*");
                startActivityForResult(intent, 10);
            }
        });

        captureBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(intent, 12);
            }
        });

        plantPageBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(getBaseContext(), PlantsActivity.class);
                startActivity(intent);
            }
        });

        predictBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                try {
                    Model model = Model.newInstance(getApplicationContext());

                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 128, 128, 3}, DataType.FLOAT32);
                    ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
                    byteBuffer.order(ByteOrder.nativeOrder());

                    int[] intValues = new int[imageSize * imageSize];

                    bitmap = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, true);
                    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
                    int pixel = 0;
                    for (int i = 0; i < imageSize; i++) {
                        for (int j = 0; j < imageSize; j++) {
                            int val = intValues[pixel++];
                            byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                            byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                            byteBuffer.putFloat((val & 0xFF) * (1.f / 255));
                        }
                    }
                    inputFeature0.loadBuffer(byteBuffer);

                    // Runs model inference and gets result.
                    Model.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    resultText.setVisibility(View.VISIBLE);
                    resultText.setText(labels[getMax(outputFeature0.getFloatArray())] + " ");

                    if(getMax(outputFeature0.getFloatArray())!=9)
                        resultBtn.setVisibility(View.VISIBLE);
                    else
                        resultBtn.setVisibility(View.INVISIBLE);
                    resultBtn.setOnClickListener(new View.OnClickListener() {
                        @Override
                        public void onClick(View view) {
                            Intent intent = new Intent(getBaseContext(), PlantActivity.class);
                            intent.putExtra("result",getMax(outputFeature0.getFloatArray()));
                            startActivity(intent);
                        }
                    });
                    // Releases model resources if no longer used.
                    model.close();
                } catch (IOException e) {
                    Toast.makeText(MainActivity.this, e.toString(), Toast.LENGTH_SHORT).show();
                    // TODO Handle the exception
                }
            }
            });
        }

        private int getMax ( float[] floatArray){
            int max = 0;
            for (int i = 0; i < floatArray.length; i++) {
                if (floatArray[i] > floatArray[max]) max = i;
            }
            return max;
        }

        void getPermission () {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                    ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.CAMERA}, 11);
                }
            }
        }

        @Override
        public void onRequestPermissionsResult ( int requestCode, @NonNull String[] permissions,
        @NonNull int[] grantResults){
            if (requestCode == 11) {
                if (grantResults.length > 0) {
                    if (grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                        this.getPermission();
                    }
                }
            }
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }

        @Override
        protected void onActivityResult ( int requestCode, int resultCode, @Nullable Intent data){
            if (requestCode == 10) {
                if (data != null) {
                    Uri uri = data.getData();
                    try {
                        bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                        imageView.setImageBitmap(bitmap);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                }
            } else if (requestCode == 12) {
                bitmap = (Bitmap) data.getExtras().get("data");
                imageView.setImageBitmap(bitmap);
            }
            super.onActivityResult(requestCode, resultCode, data);
        }
    }