import React from "react";

const EntitiesDisplay = ({ entities = [] }) => {
  return (
    <div
      style={{
        margin: "20px",
        padding: "20px",
        border: "1px solid #ccc",
        borderRadius: "5px",
      }}
    >
      <h2>Extracted Entities</h2>
      {entities.length > 0 ? (
        <ul>
          {entities.map((entity, index) => (
            <li key={index} style={{ margin: "10px 0" }}>
              <strong>{entity.label}:</strong> {entity.text}
            </li>
          ))}
        </ul>
      ) : (
        <p>No entities found.</p>
      )}
    </div>
  );
};

export default EntitiesDisplay;
