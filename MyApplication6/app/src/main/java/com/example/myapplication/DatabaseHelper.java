package com.example.myapplication;

import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

import androidx.annotation.Nullable;

public class DatabaseHelper extends SQLiteOpenHelper {

    public DatabaseHelper(@Nullable Context context) {
        super(context, "PlantData.db", null, 1);
    }

    @Override
    public void onCreate(SQLiteDatabase sqLiteDatabase) {
        sqLiteDatabase.execSQL("CREATE TABLE IF NOT EXISTS 'PlantTable' (id INTEGER PRIMARY KEY AUTOINCREMENT, plantName TEXT, plantImg TEXT, plantStatement TEXT, plantLink INTEGER)");
    }

    @Override
    public void onUpgrade(SQLiteDatabase sqLiteDatabase, int i, int i1) {
        sqLiteDatabase.execSQL("DROP TABLE IF EXISTS PlantTable");
        onCreate(sqLiteDatabase);
    }

    public Boolean insertPlantData(PlantModel plantModel){
        SQLiteDatabase sqLiteDatabase = this.getWritableDatabase();
        ContentValues contentValues = new ContentValues();
        contentValues.put("plantName",plantModel.name);
        contentValues.put("plantImg",plantModel.img);
        contentValues.put("plantStatement",plantModel.statement);
        contentValues.put("plantLink",plantModel.link);
        long result = sqLiteDatabase.insert("PlantTable",null,contentValues);
        if (result==-1){
            return false;
        }else{
            return true;
        }
    }

    public Cursor getData(){
        SQLiteDatabase sqLiteDatabase = this.getWritableDatabase();
        Cursor cursor = sqLiteDatabase.rawQuery("Select * from PlantTable",null);
        return cursor;
    }

    public Cursor getDataId(Integer id){
        SQLiteDatabase sqLiteDatabase = this.getWritableDatabase();
        Cursor cursor = sqLiteDatabase.rawQuery("Select * from PlantTable Where id = "+id,null);
        return cursor;
    }
}
