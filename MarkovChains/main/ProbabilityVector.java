public class ProbabilityVector {
    
    // Vector of probabilities.
    private double[] probabilityVector;

    /**
     * This constructor format allows the use of a string format to input the Vector which can be separated
     * by a comma by default.
     * @param columns
     * @param parsableString
     */
    public ProbabilityVector(int columns, String parsableString) throws ProbabilityVectorException {
        this.probabilityVector = new double[columns];
        if(parsableString.length() != columns){
            throw new ProbabilityVectorException("Parsable String does not have the correct number of rows and columns to make a " +
                    "completely full Vector");
        }else{
            parseString(parsableString, ",");
        }
    }

    public ProbabilityVector(int columns, String parsableString, String delimiter) throws ProbabilityVectorException {
        this.probabilityVector = new double[columns];
        if(parsableString.length() != columns){
            throw new ProbabilityVectorException("Parsable String does not have the correct number of rows and columns to make a " +
                    "completely full Vector");
        }else{
            parseString(parsableString, delimiter);
        }

        if(invalidProbabilityVector()){
            throw new ProbabilityVectorException("The vector does not accumulate to a probability of 1.0");
        }
    }

    public ProbabilityVector(int columns, double... params) throws ProbabilityVectorException {
        this.probabilityVector = new double[columns];
        if(params.length != columns){
            throw new ProbabilityVectorException("Parsable String does not have the correct number of rows and columns to make a " +
                    "completely full Vector");
        }else{
            this.probabilityVector = params;
        }

        if(invalidProbabilityVector()){
            throw new ProbabilityVectorException("The vector does not accumulate to a probability of 1.0");
        }
    }

    private void parseString(String parsableString, String delimiter){
        String[] stringArray = parsableString.split(delimiter);
        for(int column = 0; column < this.probabilityVector.length; column++){
            this.probabilityVector[column] = Double.parseDouble(stringArray[column]);
        }
        
    }

    private boolean invalidProbabilityVector(){
        double total = 0.0;
        for (double v : this.probabilityVector) {
            total += v;
        }
        return (total != 1.0);
    }

    public ProbabilityVector scalarMultiplier(int scalar) throws ProbabilityVectorException {
        double[] newValues = new double[this.probabilityVector.length];
        for(int i = 0; i < this.probabilityVector.length; i++){
            newValues[i] = scalar * this.probabilityVector[i];
        }
        return new ProbabilityVector(this.probabilityVector.length, newValues);
    }

    public int length(){
        return this.probabilityVector.length;
    }

    public double getColumnValue(int index){
        return this.probabilityVector[index];
    }
}

class ProbabilityVectorException extends Exception{
    ProbabilityVectorException(String message){
        super(message);
    }
}
