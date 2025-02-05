function DatasetSelector({ datasets, onSelect }) {
    return (
      <div>
        <select onChange={(e) => onSelect(e.target.value)}>
          <option value="">-- Select Dataset --</option>
          {datasets.map((dataset) => (
            <option key={dataset} value={dataset}>{dataset}</option>
          ))}
        </select>
      </div>
    );
  }
  
  export default DatasetSelector;
  