package com.example.myapplication;

public class PlantModel {
    int id;
    String name, statement, link, img;

    public PlantModel(String img, int id, String name, String statement, String link) {
        this.img = img;
        this.id = id;
        this.name = name;
        this.statement = statement;
        this.link = link;
    }

    public String getImg() {
        return img;
    }

    public void setImg(String img) {
        this.img = img;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getStatement() {
        return statement;
    }

    public void setStatement(String statement) {
        this.statement = statement;
    }

    public String getLink() {
        return link;
    }

    public void setLink(String link) {
        this.link = link;
    }
}
