package at.medunigraz.imi.bst.n2c2.nn;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import at.medunigraz.imi.bst.n2c2.config.Config;
import at.medunigraz.imi.bst.n2c2.util.DatasetUtil;
import org.apache.commons.io.FileUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.EvaluationBinary;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import at.medunigraz.imi.bst.n2c2.classifier.PatientBasedClassifier;
import at.medunigraz.imi.bst.n2c2.model.Criterion;
import at.medunigraz.imi.bst.n2c2.model.Eligibility;
import at.medunigraz.imi.bst.n2c2.model.Patient;

/**
 * BI-LSTM classifier for n2c2 task 2018 refactored from dl4j examples.
 * 
 * @author Markus
 *
 */
public class BILSTMC3GClassifier extends PatientBasedClassifier {

	// size of mini-batch for training
	private int miniBatchSize = 10;

	// length for truncated backpropagation through time
	private int tbpttLength = 100;

	// total number of training epochs
	private int nEpochs = 100;

	// specifies time series length
	public int truncateLength = 64;

	public int vectorSize;

	// training data
	private List<Patient> patientExamples;

	// multi layer network
	private MultiLayerNetwork net;

	// criterion index
	private Map<Criterion, Integer> criterionIndex = new HashMap<Criterion, Integer>();

	public NGramIterator fullSetIterator;

	private int trainCounter = 0;

	private static final Logger LOG = LogManager.getLogger();

	public BILSTMC3GClassifier() {

		initializeCriterionIndex();
	}

    private void initializeCriterionIndex() {
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

	public void initializeNetworkFromFile(String pathToModel) {

		// settings for memory management:
		// https://deeplearning4j.org/workspaces

		Nd4j.getMemoryManager().setAutoGcWindow(10000);
		// Nd4j.getMemoryManager().togglePeriodicGc(false);

		// instantiating generator
		fullSetIterator = new NGramIterator();

		try {

			// load a properties file
			Properties prop = new Properties();
			InputStream input = new FileInputStream(new File(pathToModel, "BILSTMC3G_MBL_0.properties"));

			prop.load(input);
			this.truncateLength = Integer.parseInt(prop.getProperty("BILSTMC3G_MBL.truncateLength.0"));

			// read char 3-grams and index
			FileInputStream fis = new FileInputStream(new File(pathToModel, "characterNGram_3_0"));
			ObjectInputStream ois = new ObjectInputStream(fis);
			ArrayList<String> characterNGram_3 = (ArrayList<String>) ois.readObject();

			fullSetIterator.characterNGram_3 = characterNGram_3;
			fullSetIterator.vectorSize = characterNGram_3.size();
			this.vectorSize = fullSetIterator.vectorSize;

			// read char 3-grams index
			fis = new FileInputStream(new File(pathToModel, "char3GramToIdxMap_0"));
			ois = new ObjectInputStream(fis);
			Map<String, Integer> char3GramToIdxMap_0 = (HashMap<String, Integer>) ois.readObject();
			fullSetIterator.char3GramToIdxMap = char3GramToIdxMap_0;

			File networkFile = new File(pathToModel, "BILSTMC3G_MBL_0.zip");

			Nd4j.getRandom().setSeed(12345);
			this.net = ModelSerializer.restoreMultiLayerNetwork(networkFile);

		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	/**
	 * SIGMOID activation and XENT loss function for binary multi-label
	 * classification.
	 */
	private void initializeNetworkBinaryMultiLabelDeep() {

		// settings for memory management:
		// https://deeplearning4j.org/workspaces

		Nd4j.getMemoryManager().setAutoGcWindow(10000);
		// Nd4j.getMemoryManager().togglePeriodicGc(false);

		try {
			fullSetIterator = new NGramIterator(patientExamples, miniBatchSize);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		vectorSize = fullSetIterator.vectorSize;
		truncateLength = fullSetIterator.maxSentences;

		int nOutFF = 150;
		int lstmLayerSize = 128;
		double l2Regulization = 0.01;
		double adaGradCore = 0.04;
		double adaGradDense = 0.01;
		double adaGradGraves = 0.008;

		// seed for reproducibility
		final int seed = 12345;
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed)
				.updater(new Adam(adaGradCore)).l2(l2Regulization)
				.weightInit(WeightInit.XAVIER).gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
				.gradientNormalizationThreshold(1.0).trainingWorkspaceMode(WorkspaceMode.ENABLED)
				.inferenceWorkspaceMode(WorkspaceMode.ENABLED).list()

				.layer(0, new DenseLayer.Builder().activation(Activation.RELU).nIn(vectorSize).nOut(nOutFF)
						.weightInit(WeightInit.RELU).updater(AdaGrad.builder().learningRate(adaGradDense).build())
						.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
						.gradientNormalizationThreshold(10).build())

				.layer(1, new DenseLayer.Builder().activation(Activation.RELU).nIn(nOutFF).nOut(nOutFF)
						.weightInit(WeightInit.RELU).updater(AdaGrad.builder().learningRate(adaGradDense).build())
						.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
						.gradientNormalizationThreshold(10).build())

				.layer(2, new DenseLayer.Builder().activation(Activation.RELU).nIn(nOutFF).nOut(nOutFF)
						.weightInit(WeightInit.RELU).updater(AdaGrad.builder().learningRate(adaGradDense).build())
						.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
						.gradientNormalizationThreshold(10).build())

				.layer(3,
						new Bidirectional(Bidirectional.Mode.ADD, new LSTM.Builder().nIn(nOutFF).nOut(lstmLayerSize)
								.updater(AdaGrad.builder().learningRate(adaGradGraves).build())
								.activation(Activation.SOFTSIGN).build()))

				.layer(4,
						new LSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
								.updater(AdaGrad.builder().learningRate(adaGradGraves).build())
								.activation(Activation.SOFTSIGN).build())

				.layer(5, new RnnOutputLayer.Builder().activation(Activation.SIGMOID)
						.lossFunction(LossFunctions.LossFunction.XENT).nIn(lstmLayerSize).nOut(13).build())

				.inputPreProcessor(0, new RnnToFeedForwardPreProcessor())
				.inputPreProcessor(3, new FeedForwardToRnnPreProcessor()).build();

		// .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)

		this.net = new MultiLayerNetwork(conf);
		this.net.init();
		this.net.setListeners(new ScoreIterationListener(1));

	}

	/**
	 * Initialize monitoring.
	 * 
	 */
	private void initializeMonitoring() {
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

    /**
	 * Training for binary multi label classifcation.
	 */
	private void trainFullSetBML() {

		// print the number of parameters in the network (and for each layer)
		Layer[] layers = net.getLayers();
		int totalNumParams = 0;
		for (int i = 0; i < layers.length; i++) {
			long nParams = layers[i].numParams();
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
				INDArray features = t.getFeatures();
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

    private void saveModel(int epoch) {
		File root = getModelDirectory(patientExamples);

		// save model after n epochs
		try {

			File locationToSave = new File(root, "BILSTMC3G_MBL_" + trainCounter + ".zip");
			boolean saveUpdater = true;
			ModelSerializer.writeModel(net, locationToSave, saveUpdater);

			// writing our character n-grams
			FileOutputStream fos = new FileOutputStream(new File(root, "characterNGram_3_" + trainCounter));
			ObjectOutputStream oos = new ObjectOutputStream(fos);
			oos.writeObject(fullSetIterator.characterNGram_3);
			oos.flush();
			oos.close();
			fos.close();

			// writing our character n-grams
			fos = new FileOutputStream(new File(root, "char3GramToIdxMap_" + trainCounter));
			oos = new ObjectOutputStream(fos);
			oos.writeObject(fullSetIterator.char3GramToIdxMap);
			oos.flush();
			oos.close();
			fos.close();

			try {
				Properties props = new Properties();
				props.setProperty("BILSTMC3G_MBL.bestModelEpoch." + trainCounter, new Integer(epoch).toString());
				props.setProperty("BILSTMC3G_MBL.truncateLength." + trainCounter,
						new Integer(truncateLength).toString());
				File f = new File(root, "BILSTMC3G_MBL_" + trainCounter + ".properties");
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

	/**
	 * Load features from narrative.
	 *
	 * @param reviewContents
	 *            Narrative content.
	 * @param maxLength
	 *            Maximum length of token series length.
	 * @return Time series feature presentation of narrative.
	 */
	private INDArray loadFeaturesForNarrative(String reviewContents, int maxLength) {

		List<String> sentences = DataUtilities.getSentences(reviewContents);

		int outputLength = Math.min(maxLength, sentences.size());
		INDArray features = Nd4j.create(1, vectorSize, outputLength);

		for (int j = 0; j < sentences.size() && j < outputLength; j++) {
			String sentence = sentences.get(j);
			INDArray vector = fullSetIterator.getChar3GramVectorToSentence(sentence);
			features.put(new INDArrayIndex[] { NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(j) },
					vector);
		}
		return features;
	}

    @Override
	public void train(List<Patient> examples) {
		if (isTrained(examples)) {
			initializeNetworkFromFile(getModelPath(examples));
		}
		else {
			this.patientExamples = examples;
			this.trainCounter = 0;

			initializeNetworkBinaryMultiLabelDeep();
//			initializeMonitoring();

			LOG.info("Minibatchsize  :\t" + miniBatchSize);
			LOG.info("tbptt length   :\t" + tbpttLength);
			LOG.info("Epochs         :\t" + nEpochs);
			LOG.info("Truncate length:\t" + truncateLength);

			trainFullSetBML();
		}
	}

	private void predict(Patient p) {
		String patientNarrative = p.getText();

		INDArray features = loadFeaturesForNarrative(patientNarrative, this.truncateLength);
		INDArray networkOutput = net.output(features);

		long timeSeriesLength = networkOutput.size(2);
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

	private static String getModelPath(List<Patient> patients) {
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
}
