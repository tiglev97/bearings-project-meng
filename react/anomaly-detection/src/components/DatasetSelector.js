function DatasetSelector({ datasets, onSelect }) {
    return (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "1.5rem", marginBottom: "2rem" }}>
        <select onChange={(e) => onSelect(e.target.value)}
          style={{ padding: "0.5rem", fontSize: "16px", borderRadius: "5px", border: "1px solid #ccc" }}>
          <option value="">-- Select Dataset --</option>
          {datasets.map((dataset) => (
            <option key={dataset} value={dataset}>{dataset}</option>
          ))}
        </select>
      </div>
    );
  }
  
  export default DatasetSelector;
  