package at.medunigraz.imi.bst.n2c2.nn;

import at.medunigraz.imi.bst.n2c2.classifier.PatientBasedClassifier;
import at.medunigraz.imi.bst.n2c2.config.Config;
import at.medunigraz.imi.bst.n2c2.model.Criterion;
import at.medunigraz.imi.bst.n2c2.model.Eligibility;
import at.medunigraz.imi.bst.n2c2.model.Patient;
import at.medunigraz.imi.bst.n2c2.util.DatasetUtil;
import org.apache.commons.io.FileUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.EvaluationBinary;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.*;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

public abstract class BaseNNClassifier extends PatientBasedClassifier {

    private static final Logger LOG = LogManager.getLogger();

    // size of mini-batch for training
    protected int miniBatchSize = 10;

    // length for truncated backpropagation through time
    protected int tbpttLength = 100;

    // total number of training epochs
    protected int nEpochs = 100;

    // specifies time series length
    protected int truncateLength = 64;

    public int vectorSize;

    // training data
    protected List<Patient> patientExamples;

    // multi layer network
    protected MultiLayerNetwork net;

    // criterion index
    protected Map<Criterion, Integer> criterionIndex = new HashMap<Criterion, Integer>();

    public DataSetIterator fullSetIterator;

    // training counter
    protected int trainCounter = 0;

    protected void initializeCriterionIndex() {
        this.criterionIndex.put(Criterion.ABDOMINAL, 0);
        this.criterionIndex.put(Criterion.ADVANCED_CAD, 1);
        this.criterionIndex.put(Criterion.ALCOHOL_ABUSE, 2);
        this.criterionIndex.put(Criterion.ASP_FOR_MI, 3);
        this.criterionIndex.put(Criterion.CREATININE, 4);
        this.criterionIndex.put(Criterion.DIETSUPP_2MOS, 5);
        this.criterionIndex.put(Criterion.DRUG_ABUSE, 6);
        this.criterionIndex.put(Criterion.ENGLISH, 7);
        this.criterionIndex.put(Criterion.HBA1C, 8);
        this.criterionIndex.put(Criterion.KETO_1YR, 9);
        this.criterionIndex.put(Criterion.MAJOR_DIABETES, 10);
        this.criterionIndex.put(Criterion.MAKES_DECISIONS, 11);
        this.criterionIndex.put(Criterion.MI_6MOS, 12);
    }

    /**
     * Training for binary multi label classifcation.
     */
    protected void trainFullSetBML() {

        // print the number of parameters in the network (and for each layer)
        Layer[] layers = net.getLayers();
        int totalNumParams = 0;
        for (int i = 0; i < layers.length; i++) {
            int nParams = layers[i].numParams();
            LOG.info("Number of parameters in layer " + i + ": " + nParams);
            totalNumParams += nParams;
        }
        LOG.info("Total number of network parameters: " + totalNumParams);

        int epochCounter = 0;

        EvaluationBinary eb = new EvaluationBinary();
        do {

            EvaluationBinary ebepoch = new EvaluationBinary();

            net.fit(fullSetIterator);
            fullSetIterator.reset();

            LOG.info("Epoch " + epochCounter++ + " complete.");
            LOG.info("Starting FULL SET evaluation:");

            while (fullSetIterator.hasNext()) {
                DataSet t = fullSetIterator.next();
                INDArray features = t.getFeatureMatrix();
                INDArray lables = t.getLabels();
                INDArray inMask = t.getFeaturesMaskArray();
                INDArray outMask = t.getLabelsMaskArray();
                INDArray predicted = net.output(features, false, inMask, outMask);

                ebepoch.eval(lables, predicted, outMask);
                eb = ebepoch;
            }

            fullSetIterator.reset();
            LOG.info(System.getProperty("line.separator") + ebepoch.stats());
            LOG.info("Average accuracy: {}", eb.averageAccuracy());

        } while (eb.averageAccuracy() < 0.95);

        // save model and parameters for reloading
        this.saveModel(epochCounter);

        trainCounter++;
    }

    /**
     * Initialize monitoring.
     *
     */
    protected void initializeMonitoring() {
        // setting monitor
        UIServer uiServer = UIServer.getInstance();

        // configure where the network information (gradients, score vs. time
        // etc) is to be stored. Here: store in memory.
        // Alternative: new FileStatsStorage(File), for saving and loading later
        StatsStorage statsStorage = new InMemoryStatsStorage();

        // Attach the StatsStorage instance to the UI: this allows the contents
        // of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        // then add the StatsListener to collect this information from the
        // network, as it trains
        net.setListeners(new StatsListener(statsStorage));
    }

    protected void saveModel(int epoch) {
        File root = getModelDirectory(patientExamples);

        // save model after n epochs
        try {

            File locationToSave = new File(root, getModelName() + "_" + trainCounter + ".zip");
            boolean saveUpdater = true;
            ModelSerializer.writeModel(net, locationToSave, saveUpdater);

            try {
                Properties props = new Properties();
                props.setProperty(getModelName() + ".bestModelEpoch." + trainCounter, new Integer(epoch).toString());
                props.setProperty(getModelName() + ".truncateLength." + trainCounter, new Integer(truncateLength).toString());
                File f = new File(root, getModelName() + "_" + trainCounter + ".properties");
                OutputStream out = new FileOutputStream(f);
                props.store(out, "Best model at epoch");
            } catch (Exception e) {
                e.printStackTrace();
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    protected abstract String getModelName();

    private void predict(Patient p) {
        String patientNarrative = p.getText();

        INDArray features = loadFeaturesForNarrative(patientNarrative, this.truncateLength);
        INDArray networkOutput = net.output(features);

        int timeSeriesLength = networkOutput.size(2);
        INDArray probabilitiesAtLastWord = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeSeriesLength - 1));

        criterionIndex.forEach((c, idx) -> {
            double probabilityForCriterion = probabilitiesAtLastWord.getDouble(criterionIndex.get(c));
            Eligibility eligibility = probabilityForCriterion > 0.5 ? Eligibility.MET : Eligibility.NOT_MET;

            p.withCriterion(c, eligibility);

            LOG.info("\n\n-------------------------------");
            LOG.info("Patient: " + p.getID());
            LOG.info("Probabilities at last time step for {}", c.name());
            LOG.info("Probability\t" + c.name() + ": " + probabilityForCriterion);
            LOG.info("Eligibility\t" + c.name() + ": " + eligibility.name());
        });
    }

    /*
     * (non-Javadoc)
     *
     * @see
     * at.medunigraz.imi.bst.n2c2.classifier.Classifier#predict(at.medunigraz.
     * imi.bst.n2c2.model.Patient, at.medunigraz.imi.bst.n2c2.model.Criterion)
     */
    @Override
    public Eligibility predict(Patient p, Criterion c) {
        predict(p);
        return p.getEligibility(c);
    }

    @Override
    public void train(List<Patient> examples) {
        if (isTrained(examples)) {
            initializeNetworkFromFile(getModelPath(examples));
        }
        else {
            this.patientExamples = examples;
            this.trainCounter = 0;

            initializeNetwork();
//			initializeMonitoring();

            LOG.info("Minibatchsize  :\t" + miniBatchSize);
            LOG.info("tbptt length   :\t" + tbpttLength);
            LOG.info("Epochs         :\t" + nEpochs);
            LOG.info("Truncate length:\t" + truncateLength);

            trainFullSetBML();
        }
    }

    protected static String getModelPath(List<Patient> patients) {
        return Config.NN_MODELS + File.separator + DatasetUtil.getChecksum(patients) + File.separator;
    }

    public File getModelDirectory(List<Patient> patients) {
        File modelDir = new File(getModelPath(patients));
        modelDir.mkdirs();
        return modelDir;
    }

    public void deleteModelDir(List<Patient> patients) throws IOException {
        FileUtils.deleteDirectory(getModelDirectory(patients));
    }

    public boolean isTrained(List<Patient> patients) {
        return new File(getModelPath(patients)).isDirectory();
    }

    protected abstract INDArray loadFeaturesForNarrative(String reviewContents, int maxLength);

    public void initializeNetworkFromFile(String pathToModel) {
        try {
            File networkFile = new File(pathToModel, getModelName() + "_0.zip");
            this.net = ModelSerializer.restoreMultiLayerNetwork(networkFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    protected abstract void initializeNetwork();
}