package com.example.myapplication;


import android.content.ClipData;
import android.content.Context;
import android.content.Intent;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;

public class RecyclerContactAdapter extends RecyclerView.Adapter<RecyclerContactAdapter.ViewHolder> {

    private ArrayList<PlantModel> plantModels;
    private RecyclerViewClickListener listener;
    private Context context;

    public RecyclerContactAdapter(ArrayList<PlantModel> plantModels,RecyclerViewClickListener listener,Context context){
        this.plantModels = plantModels;
        this.listener = listener;
        this.context = context;
    }

    @NonNull
    @Override
    public RecyclerContactAdapter.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.plant_card,parent,false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        holder.imageView.setImageResource(context.getResources().getIdentifier(plantModels.get(position).img,"drawable", context.getPackageName()));
        holder.txtName.setText(plantModels.get(position).name);
        holder.txtContact.setText(plantModels.get(position).statement.substring(0,50)+"...");
    }



    @Override
    public int getItemCount() {
        return plantModels.size();
    }

    public interface RecyclerViewClickListener{
        void onClick(View v,int position);
    }

    public class ViewHolder extends RecyclerView.ViewHolder implements  View.OnClickListener{
        TextView txtName, txtContact;
        ImageView imageView;

        public ViewHolder(final View itemView) {
            super(itemView);
            txtContact = itemView.findViewById(R.id.txtContact);
            txtName = itemView.findViewById(R.id.txtName);
            imageView = itemView.findViewById(R.id.imgContact);
            itemView.setOnClickListener(this);
        }

        @Override
        public void onClick(View view) {
            listener.onClick(view,getAdapterPosition());
        }
    }
}
