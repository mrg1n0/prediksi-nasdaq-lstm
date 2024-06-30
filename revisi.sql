-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Jun 30, 2024 at 10:13 PM
-- Server version: 10.4.28-MariaDB
-- PHP Version: 8.1.17

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `revisi`
--

-- --------------------------------------------------------

--
-- Table structure for table `result`
--

CREATE TABLE `result` (
  `id` int(11) NOT NULL,
  `username` varchar(255) NOT NULL,
  `stock_code` varchar(10) NOT NULL,
  `rentan_waktu` varchar(20) NOT NULL,
  `jumlah_data` int(11) NOT NULL,
  `kolom_prediksi` varchar(50) NOT NULL,
  `persentase_train_size` int(11) NOT NULL,
  `epochs` int(11) NOT NULL,
  `batch_size` int(11) NOT NULL,
  `rmse` float NOT NULL,
  `mape` float NOT NULL,
  `rsquared` float NOT NULL,
  `date` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `result`
--

INSERT INTO `result` (`id`, `username`, `stock_code`, `rentan_waktu`, `jumlah_data`, `kolom_prediksi`, `persentase_train_size`, `epochs`, `batch_size`, `rmse`, `mape`, `rsquared`, `date`) VALUES
(8, 'raihan', 'AAPL', '2019-2024', 1825, 'Close', 80, 100, 25, 2.63856, 1.11689, 0.971845, '2024-06-19 21:53:42'),
(9, 'raihan', 'GOOGL', '2019-2024', 1825, 'Close', 80, 50, 25, 2.68452, 1.49602, 0.985742, '2024-06-19 22:00:37'),
(10, 'raihan', 'AMZN', '2019-2024', 1825, 'Close', 80, 50, 25, 2.63023, 1.50387, 0.992149, '2024-06-19 22:03:15'),
(11, 'raihan', 'AMZN', '2019-2024', 1825, 'Close', 80, 50, 25, 3.1161, 1.77107, 0.988981, '2024-06-19 22:06:55'),
(12, 'raihan', 'AMZN', '2019-2024', 1825, 'Close', 80, 50, 25, 2.71795, 1.54432, 0.991617, '2024-06-19 22:17:12'),
(13, 'raihan', 'AMZN', '2019-2024', 1825, 'Close', 80, 50, 25, 2.82457, 1.61321, 0.990947, '2024-06-19 22:18:18'),
(14, 'raihan', 'AMZN', '2019-2024', 1825, 'Close', 80, 50, 25, 2.62948, 1.50424, 0.992154, '2024-06-19 22:24:25'),
(15, 'raihan', 'TSLA', '2019-2024', 1825, 'Open', 80, 100, 15, 4.62483, 1.62464, 0.985861, '2024-06-20 07:53:58'),
(16, 'user', 'AAPL', '2019-2024', 1825, 'Close', 80, 100, 25, 2.80485, 1.17832, 0.965935, '2024-06-30 19:39:52');

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `id` int(11) NOT NULL,
  `username` varchar(255) NOT NULL,
  `password` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `users`
--

INSERT INTO `users` (`id`, `username`, `password`) VALUES
(3, 'user', '$2b$12$ZAjgtt3Nom3tS2nEAFbN2O9dge8u0FUdNPT84elgGcGj06XQEICJ6'),
(5, 'raihan', '$2b$12$P31k5purfsCEhs2HeiCDpeG1JqUZJI5bCn.K2E64GD3Tf4IwrNXj.');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `result`
--
ALTER TABLE `result`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `result`
--
ALTER TABLE `result`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=17;

--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
