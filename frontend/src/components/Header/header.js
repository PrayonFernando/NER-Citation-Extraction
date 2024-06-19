import React from "react";
import { TextField, Grid } from "@mui/material";
import "./header.css";
import court1 from "../../assets/court1.jpg";

function Home() {
  return (
    <Grid
      container
      style={{
        backgroundColor: "#003399",
        height: "10vh",
        padding: "20px",
        alignContent: "center",
      }}
    >
      <header
        className="header"
        style={{
          marginLeft: "43%",
        }}
      >
        <Grid container item direction={"row"}>
          <img
            src={court1}
            alt="Logo"
            style={{ marginRight: "8px", height: "11vh" }}
          />
          <h1>JuristMind</h1>
        </Grid>
      </header>
    </Grid>
  );
}

export default Home;
