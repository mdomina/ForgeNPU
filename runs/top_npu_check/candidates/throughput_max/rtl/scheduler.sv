module scheduler #(
  parameter int DATA_WIDTH = 8,
  parameter int ROWS = 2,
  parameter int COLS = 2,
  parameter int DEPTH = 4,
  parameter int ADDR_WIDTH = $clog2(DEPTH),
  parameter int MAX_DIM = (ROWS > COLS) ? ROWS : COLS
) (
  input  logic clk,
  input  logic rst_n,
  input  logic start_i,
  input  logic signed [ROWS*DATA_WIDTH-1:0] activation_slot0_i,
  input  logic signed [ROWS*DATA_WIDTH-1:0] activation_slot1_i,
  input  logic signed [COLS*DATA_WIDTH-1:0] weight_slot0_i,
  input  logic signed [COLS*DATA_WIDTH-1:0] weight_slot1_i,
  output logic dma_valid_o,
  output logic dma_write_weights_o,
  output logic [ADDR_WIDTH-1:0] dma_addr_o,
  output logic signed [MAX_DIM*DATA_WIDTH-1:0] dma_payload_o,
  output logic load_vector_en_o,
  output logic [ADDR_WIDTH-1:0] activation_read_addr_o,
  output logic [ADDR_WIDTH-1:0] weight_read_addr_o,
  output logic compute_en_o,
  output logic clear_acc_o,
  output logic busy_o,
  output logic done_o,
  output logic [3:0] state_o
);
  localparam logic [3:0] S_IDLE = 4'd0;
  localparam logic [3:0] S_DMA_ACT0 = 4'd1;
  localparam logic [3:0] S_DMA_WGT0 = 4'd2;
  localparam logic [3:0] S_DMA_ACT1 = 4'd3;
  localparam logic [3:0] S_DMA_WGT1 = 4'd4;
  localparam logic [3:0] S_LOAD0 = 4'd5;
  localparam logic [3:0] S_LOAD1 = 4'd6;
  localparam logic [3:0] S_COMPUTE0 = 4'd7;
  localparam logic [3:0] S_COMPUTE1 = 4'd8;
  localparam logic [3:0] S_CLEAR = 4'd9;
  localparam logic [3:0] S_DONE = 4'd10;
  localparam logic [ADDR_WIDTH-1:0] ADDR_ZERO = '0;
  localparam logic [ADDR_WIDTH-1:0] ADDR_ONE = 1;

  logic [3:0] state_q;
  logic [3:0] state_d;

  always_comb begin
    state_d = state_q;
    unique case (state_q)
      S_IDLE: begin
        if (start_i) begin
          state_d = S_DMA_ACT0;
        end
      end
      S_DMA_ACT0: state_d = S_DMA_WGT0;
      S_DMA_WGT0: state_d = S_DMA_ACT1;
      S_DMA_ACT1: state_d = S_DMA_WGT1;
      S_DMA_WGT1: state_d = S_LOAD0;
      S_LOAD0: state_d = S_LOAD1;
      S_LOAD1: state_d = S_COMPUTE0;
      S_COMPUTE0: state_d = S_COMPUTE1;
      S_COMPUTE1: state_d = S_CLEAR;
      S_CLEAR: state_d = S_DONE;
      S_DONE: state_d = S_IDLE;
      default: state_d = S_IDLE;
    endcase
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state_q <= S_IDLE;
    end else begin
      state_q <= state_d;
    end
  end

  always_comb begin
    dma_valid_o = 1'b0;
    dma_write_weights_o = 1'b0;
    dma_addr_o = ADDR_ZERO;
    dma_payload_o = '0;
    load_vector_en_o = 1'b0;
    activation_read_addr_o = ADDR_ZERO;
    weight_read_addr_o = ADDR_ZERO;
    compute_en_o = 1'b0;
    clear_acc_o = 1'b0;
    busy_o = (state_q != S_IDLE) && (state_q != S_DONE);
    done_o = (state_q == S_DONE);
    state_o = state_q;

    unique case (state_q)
      S_DMA_ACT0: begin
        dma_valid_o = 1'b1;
        dma_addr_o = ADDR_ZERO;
        dma_payload_o[0 +: ROWS*DATA_WIDTH] = activation_slot0_i;
      end
      S_DMA_WGT0: begin
        dma_valid_o = 1'b1;
        dma_write_weights_o = 1'b1;
        dma_addr_o = ADDR_ZERO;
        dma_payload_o[0 +: COLS*DATA_WIDTH] = weight_slot0_i;
      end
      S_DMA_ACT1: begin
        dma_valid_o = 1'b1;
        dma_addr_o = ADDR_ONE;
        dma_payload_o[0 +: ROWS*DATA_WIDTH] = activation_slot1_i;
      end
      S_DMA_WGT1: begin
        dma_valid_o = 1'b1;
        dma_write_weights_o = 1'b1;
        dma_addr_o = ADDR_ONE;
        dma_payload_o[0 +: COLS*DATA_WIDTH] = weight_slot1_i;
      end
      S_LOAD0: begin
        load_vector_en_o = 1'b1;
        activation_read_addr_o = ADDR_ZERO;
        weight_read_addr_o = ADDR_ZERO;
      end
      S_LOAD1: begin
        load_vector_en_o = 1'b1;
        activation_read_addr_o = ADDR_ONE;
        weight_read_addr_o = ADDR_ONE;
      end
      S_COMPUTE0,
      S_COMPUTE1: begin
        compute_en_o = 1'b1;
      end
      S_CLEAR: begin
        clear_acc_o = 1'b1;
      end
      default: begin
      end
    endcase
  end
endmodule
